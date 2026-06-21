import numpy as np, joblib
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

CLASSES = ["bellflower","daisy","dandelion","lotus","rose","sunflower","tulip"]
IMG=(32,32); DIR="flower-training"
X,y=[],[]
for lab,c in enumerate(CLASSES):
    for f in os.listdir(os.path.join(DIR,c)):
        if f.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp")):
            try:
                im=Image.open(os.path.join(DIR,c,f)).convert("RGB").resize(IMG)
                X.append(np.array(im)); y.append(lab)
            except: pass
X=np.array(X,dtype="float64"); y=np.array(y)
print("data",X.shape)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
Xtr=Xtr.reshape(len(Xtr),-1); Xte=Xte.reshape(len(Xte),-1)
mean=np.mean(Xtr,axis=0); Xtr-=mean; Xte-=mean
sc=StandardScaler(); Xtr=sc.fit_transform(Xtr); Xte=sc.transform(Xte)
clf=LinearSVC(C=0.001,class_weight="balanced",dual=False,max_iter=10000,random_state=42)
clf.fit(Xtr,ytr)
print("test acc",round(accuracy_score(yte,clf.predict(Xte)),4))
joblib.dump({"model":clf,"classes":CLASSES,"img_size":IMG,"mean_image":mean,
             "scaler":sc,"model_type":"LinearSVC"},"best_svm_flower.joblib")
print("SAVED best_svm_flower.joblib (7 lop)")
