import luigi
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
import pickle

# Define unction to calculate Euclidean distance between tweet geolocation and all other cities in cities.csv
def find_closest_city(coord,dataframe):
    distance = np.sqrt(((dataframe[['latitude','longitude']] - coord) ** 2).sum(1))
    return dataframe[['name']].iloc[(distance).idxmin()][0]


class CleanDataTask(luigi.Task):
    """ Cleans the input CSV file by removing any rows without valid geo-coordinates.

        Output file should contain just the rows that have geo-coordinates and
        non-(0.0, 0.0) files.
    """
    tweet_file = luigi.Parameter()
    cities_file = luigi.Parameter(default='cities.csv')
    output_file = luigi.Parameter(default='clean_data.csv')

    def output(self):
        return luigi.LocalTarget(self.output_file)
    
    def run(self):
        tweets_df = pd.read_csv(self.tweet_file, encoding = "ISO-8859-1")
        cities_df = pd.read_csv(self.cities_file, encoding = "ISO-8859-1")
 
        # drop missing values and values with no tweet coordinate
        tweets_clean_df = tweets_df[['airline_sentiment', 'tweet_coord']].dropna()
        tweets_clean_df = tweets_clean_df.loc[tweets_clean_df['tweet_coord'] != '[0.0, 0.0]']
        
        # preprocessing to convert tweet coord to float for distance calculation
        ls = [row.strip('][').split(',') for row in tweets_clean_df.tweet_coord]
        ls2 = [[float(x) for x in row] for row in ls ]
        
        # Find nearest city to each tweet geolocation based on Euclidean distance
        tweets_clean_df['nearest_city']=[find_closest_city(x, cities_df) for x in ls2]
        
        # replace airline_sentiment with numerical labels for classification
        numerical_labels = {'negative': 0, 'neutral': 1, 'positive': 2}
        tweets_clean_df.replace({'airline_sentiment': numerical_labels}, inplace=True)
        
        # save cleaned dataFrame
        tweets_clean_df.to_csv(self.output().path)        
    


class TrainingDataTask(luigi.Task):
    """ Extracts features/outcome variable in preparation for training a model.

        Output file should have columns corresponding to the training data:
        - y = airline_sentiment (coded as 0=negative, 1=neutral, 2=positive)
        - X = a one-hot coded column for each city in "cities.csv"
    """
    tweet_file = luigi.Parameter()
    output_file = luigi.Parameter(default='features.csv')

    def requires(self):
        return CleanDataTask(tweet_file=self.tweet_file)
    
    def output(self):
        return luigi.LocalTarget(self.output_file)
    
    def run(self):
        tweets_clean_df = pd.read_csv(self.input().open('r'), encoding = "ISO-8859-1")
        
        # one-hot encode categorical features defined by nearest city to tweet geo-location
        X = pd.get_dummies(tweets_clean_df.nearest_city)
        y = tweets_clean_df.airline_sentiment
        features_df = pd.concat([X,y], axis=1)
        
        # save features of X,y to csv for next stage of pipeline
        features_df.to_csv(self.output().path)
        
        

class TrainModelTask(luigi.Task):
    """ Trains a classifier to predict negative, neutral, positive
        based only on the input city.

        Output file should be the pickle'd model.
    """
    tweet_file = luigi.Parameter()
    output_file = luigi.Parameter(default='model.pkl')
    
    def requires(self):
        return TrainingDataTask(tweet_file=self.tweet_file), CleanDataTask(tweet_file=self.tweet_file)
    
    def output(self):
        return luigi.LocalTarget(self.output_file)
    
    def run(self):
        # load features dataframe and pull out X,y
        features_df = pd.read_csv(self.input()[0].open('r'), index_col=0)
        tweets_clean_df = pd.read_csv(self.input()[1].open('r'), index_col=0, encoding = "ISO-8859-1")
        
        # create train and test set with sklearn 
        X = features_df.drop(['airline_sentiment'], axis=1)
        y = features_df.airline_sentiment
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
        
        # oversample minority class using SMOTE because of class imbalance of sentiment
        sm = SMOTE()
        
        # random forest classifier using grid search
        rf = RandomForestClassifier()
        
        # setup pipeline so that oversampling can be done for each fold to prevent data leakage
        pipeline = Pipeline([('sm', sm), ('rf', rf)])
        
        # parameter grid to search over
        params = {'rf__max_depth' : list(range(2,5)),
                  'rf__max_features' : ['auto','sqrt'],
                  'rf__bootstrap' : [True, False],
                  'rf__n_estimators' : [10, 20, 50, 100]
                  }
        # run grid search rfc with 5-fold cross-validation
        rf_grid = GridSearchCV(pipeline, params, cv = 5)
        rf_grid.fit(X_train, y_train)

        print("\n-Model Summary-\n")
        # Print the best parameters and highest score
        print("Best parameters found: ", rf_grid.best_params_)
        print("Highest score found: ", rf_grid.best_score_)
        
        print('Train score: %0.4f' % rf_grid.best_estimator_.score(X_train, y_train))
        print('Test score: %0.4f' % rf_grid.best_estimator_.score(X_test, y_test))
        print('F1 Score (Macro Avg): %0.4f' % f1_score(y_test, rf_grid.best_estimator_.predict(X_test), average='macro'))
        
        print("\n-Classification Report-\n")
        print(classification_report(y_test, rf_grid.best_estimator_.predict(X_test)))
        
        # View a list of the features and their importance scores
        print("\n-Feature Importances-\n")
        lis=list(zip(tweets_clean_df.nearest_city, rf_grid.best_estimator_.named_steps["rf"].feature_importances_))
        print(sorted(lis, key=lambda t: t[1], reverse=True)[:15])
        
        with open(self.output().path, 'wb') as f:
            pickle.dump(rf_grid.best_estimator_, f)
        

class ScoreTask(luigi.Task):
    """ Uses the scored model to compute the sentiment for each city.

        Output file should be a four column CSV with columns:
        - city name
        - negative probability
        - neutral probability
        - positive probability
    """
    tweet_file = luigi.Parameter()
    output_file = luigi.Parameter(default='scores.csv')

    def requires(self):
        return TrainModelTask(tweet_file=self.tweet_file), TrainingDataTask(tweet_file=self.tweet_file), CleanDataTask(tweet_file=self.tweet_file)
    
    def output(self):
        return luigi.LocalTarget(self.output_file)
    
    def run(self):
        #Load the trained classifier model
        rfc_model = pickle.load(open(self.input()[0].path, 'rb'))
        # Load features and cleaned dataframe
        features_df = pd.read_csv(self.input()[1].open('r'), index_col=0)
        tweets_clean_df = pd.read_csv(self.input()[2].open('r'), index_col=0, encoding = "ISO-8859-1")
        X = features_df.drop(['airline_sentiment'], axis=1)
        
        # generate probabilities for each sentiment using trained model
        prob = rfc_model.predict_proba(X)
        
        # store probabilities in dataframe for output from pipeline
        scores_df = pd.DataFrame(tweets_clean_df['nearest_city'])
        scores_df['negative probability'] = list(prob[:, 0])
        scores_df['neutral probability'] = list(prob[:, 1])
        scores_df['positive probability'] = list(prob[:, 2])
        scores_df_sorted = scores_df.sort_values(by=['positive probability'], ascending=False)
        
        # get unique city names
        scores_df_sorted = scores_df_sorted.drop_duplicates()
        
        # save sorted list of cities by predicted positive sentiment
        scores_df_sorted.to_csv(self.output().path, index=False)
        

if __name__ == "__main__":
    luigi.run()
