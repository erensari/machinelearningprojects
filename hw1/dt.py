
class DecisionTreeClassifier:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.tree = None

    def gini_impurity(self, y):
        # Gini Saflığını hesaplar
        m = len(y)
        p = [sum([1 for i in y if i == c]) / m for c in set(y)]
        return 1 - sum([pk ** 2 for pk in p])


    def best_split(self, X, y):
        m, n = len(X), len(X[0])
        best_gini = float('inf')
        best_index, best_value = None, None

        for index in range(n):
            values = set(row[index] for row in X)
            for value in values:
                # Split veriyi left ve right olarak ikiye ayır
                left_y = [y[i] for i in range(m) if X[i][index] <= value]
                right_y = [y[i] for i in range(m) if X[i][index] > value]

                # Gini impurity'yi hesapla
                gini_left = self.gini_impurity(left_y)
                gini_right = self.gini_impurity(right_y)
                gini = (len(left_y) / m) * gini_left + (len(right_y) / m) * gini_right

                # En düşük Gini impurity'yi güncelle
                if gini < best_gini:
                    best_gini = gini
                    best_index = index
                    best_value = value

        return best_index, best_value


    def build_tree(self, X, y, depth):
        # Yaprak düğüm veya maks. derinliğe ulaşıldıysa, yaprak düğüm olarak dön
        if depth >= self.max_depth or len(set(y)) == 1:
            most_common_class = max(set(y), key=y.count)
            node = {
                'type': 'leaf',
                'class': most_common_class,
                'depth': depth,
                'samples': len(y)
            }
            return node

        # En iyi bölmeyi bul
        best_index, best_value = self.best_split(X, y)
        if best_index is None:
            most_common_class = max(set(y), key=y.count)
            node = {
                'type': 'leaf',
                'class': most_common_class,
                'depth': depth,
                'samples': len(y)
            }
            return node

        # Veri setini en iyi bölmeye göre ikiye ayır
        left_indices = [i for i in range(len(X)) if X[i][best_index] <= best_value]
        right_indices = [i for i in range(len(X)) if X[i][best_index] > best_value]

        left_X = [X[i] for i in left_indices]
        left_y = [y[i] for i in left_indices]
        right_X = [X[i] for i in right_indices]
        right_y = [y[i] for i in right_indices]

        # İç düğüm oluştur
        node = {
            'type': 'node',
            'index': best_index,
            'value': best_value,
            'depth': depth,
            'samples': len(y),
            'left': self.build_tree(left_X, left_y, depth + 1),
            'right': self.build_tree(right_X, right_y, depth + 1)
        }

        return node





    def fit(self, X, y):
        self.tree = self.build_tree(X, y, 0)



    def predict_one(self, x, node):
    
        while node['type'] != 'leaf':
            if x[node['index']] <= node['value']:
                node = node['left']
            else:
                node = node['right']
        return node['class']

    def predict(self, X):
    
        predictions = [self.predict_one(x, self.tree) for x in X]
        return predictions







    






'''
    def best_split(self, X, y):
        # En iyi split'i bulur
        m, n = len(X), len(X[0])
        best_gini = float('inf')
        best_index, best_value = None, None

        for index in range(n):
            values = set([row[index] for row in X])
            for value in values:
                left_y = [y[i] for i in range(m) if X[i][index] <= value]
                right_y = [y[i] for i in range(m) if X[i][index] > value]
                
                gini_left = self.gini_impurity(left_y)
                gini_right = self.gini_impurity(right_y)
                gini = (len(left_y) / m) * gini_left + (len(right_y) / m) * gini_right

                if gini < best_gini:
                    best_gini = gini
                    best_index = index
                    best_value = value

        return best_index, best_value


    
    def build_tree(self, X, y, depth):
        if depth >= self.max_depth or len(set(y)) == 1:
            return {'type': 'leaf', 'class': max(set(y), key=y.count)}

        best_index, best_value = self.best_split(X, y)
        if best_index is None:
            return {'type': 'leaf', 'class': max(set(y), key=y.count)}

        left_X = [X[i] for i in range(len(X)) if X[i][best_index] <= best_value]
        left_y = [y[i] for i in range(len(X)) if X[i][best_index] <= best_value]
        right_X = [X[i] for i in range(len(X)) if X[i][best_index] > best_value]
        right_y = [y[i] for i in range(len(X)) if X[i][best_index] > best_value]

        return {
            'type': 'node',
            'index': best_index,
            'value': best_value,
            'left': self.build_tree(left_X, left_y, depth + 1),
            'right': self.build_tree(right_X, right_y, depth + 1)
        }


        

        def predict_one(self, x, node):
        if node['type'] == 'leaf':
            return node['class']
        if x[node['index']] <= node['value']:
            return self.predict_one(x, node['left'])
        else:
            return self.predict_one(x, node['right'])

    def predict(self, X):
        return [self.predict_one(x, self.tree) for x in X]
    




 '''