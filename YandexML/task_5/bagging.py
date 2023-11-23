import numpy as np

class SimplifiedBaggingRegressor:
    def __init__(self, num_bags, oob=False):
        self.num_bags = num_bags
        self.oob = oob
        
        
    def _generate_splits(self, data: np.ndarray):
        '''
        Generate indices for every bag and store in self.indices_list list
        '''
        self.indices_list = []
        data_length = len(data)
        for bag in range(self.num_bags):
            a = np.random.randint(data_length, size=data_length)
            self.indices_list.append(a)
        
    def fit(self, model_constructor, data, target):
        '''
        Fit model on every bag.
        Model constructor with no parameters (and with no ()) is passed to this function.
        
        example:
        
        bagging_regressor = SimplifiedBaggingRegressor(num_bags=10, oob=True)
        bagging_regressor.fit(LinearRegression, X, y)
        '''
        self.data = None
        self.target = None
        self._generate_splits(data)
        assert len(set(list(map(len, self.indices_list)))) == 1, 'All bags should be of the same length!'
        assert list(map(len, self.indices_list))[0] == len(data), 'All bags should contain `len(data)` number of elements!'
        self.models_list = []
        for bag in range(self.num_bags):
            model = model_constructor()
            data_bag, target_bag = data[self.indices_list[bag]], target[self.indices_list[bag]] # Your Code Here
            self.models_list.append(model.fit(data_bag, target_bag)) # store fitted models here
        if self.oob:
            self.data = data
            self.target = target
        
        self._get_oob_predictions_from_every_model()
        
    def predict(self, data):
        '''
        Get average prediction for every object from passed dataset
        '''
        preds = []
        for model in self.models_list:
            preds.append(model.predict(data))

        return np.mean(preds, axis=0)
    
    def _get_oob_predictions_from_every_model(self):
        '''
        Generates list of lists, where list i contains predictions for self.data[i] object
        from all models, which have not seen this object during training phase
        '''
        list_of_predictions_lists = [[] for _ in range(len(self.data))]

        for i in range(len(self.data)):
            for (bag, model) in zip(self.indices_list, self.models_list):
                oob_indexes = np.setdiff1d(range(len(self.data)), bag)
                if i in oob_indexes:
                    list_of_predictions_lists[i].append(model.predict([self.data[i]]))

        self.list_of_predictions_lists = np.array(list_of_predictions_lists, dtype=object)

    
    def _get_averaged_oob_predictions(self):
        '''
        Compute average prediction for every object from training set.
        If object has been used in all bags on training phase, return None instead of prediction
        '''
        
        preds = np.zeros(len(self.data))
        preds_count = np.zeros(len(self.data))
        mask = np.zeros(len(self.data))

        for (bag, model) in zip(self.indices_list, self.models_list):
            oob_indexes = np.setdiff1d(range(len(self.data)), bag)
            pred = model.predict(self.data[oob_indexes])
            preds[oob_indexes] += pred
            preds_count[oob_indexes] += 1
            mask[oob_indexes] = 1
        
        self.mask = mask == 1

        self.oob_predictions = np.divide(preds[self.mask], preds_count[self.mask]) # Your Code Here
        
        
    def OOB_score(self):
        '''
        Compute mean square error for all objects, which have at least one prediction
        '''
        self._get_averaged_oob_predictions()
        dif = self.oob_predictions - self.target[self.mask]
        return np.sum(np.multiply(dif, dif))
    