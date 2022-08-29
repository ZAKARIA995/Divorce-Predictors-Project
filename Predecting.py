
import pandas as pd

PATH = 'divorce.xlsx'

dataset = pd.read_excel(PATH)


X = dataset.iloc[:,:-1].values

Y = dataset.iloc[:,-1].values


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.25 , random_state = 0)



from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
Y_train = scaler.fit_transform(Y_train.reshape(-1,1))
Y_test =scaler.fit_transform(Y_test.reshape(-1,1))

from sklearn.neighbors import KNeighborsClassifier
Classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p=2)
Classifier.fit(X_train, Y_train)
Y_Pred = Classifier.predict(X_test)



from sklearn.metrics import confusion_matrix,classification_report
hm = confusion_matrix(Y_test, Y_Pred)
print(classification_report(Y_test, Y_Pred))


