import json


data = json.load(open('PR_epoch_18.json'))




best_score = 0.0
best_thr = None
best_P = 0.0
best_R = 0.0

for dt in list(data.values())[:-1]:

    P, R, thr = dt
    F_score = (P**5) * R
    if F_score > best_score:
        best_score = F_score
        best_thr = thr
        best_P = P
        best_R = R
print('asd')