import joblib
 
model=joblib.load('salary_predict.pk1')
inp=int(input("Enter year of experience of employee: "))

out=int(model.predict([[inp]]))
print(f"according to model employee should get: {out}")