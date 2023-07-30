import csv
from flask import Flask, request
import os
from threading import Thread


class ScoreBoard:
    def __init__(self, autoload: bool = True):
        self.scoreboard = []
        self.membercount = 0
        self.headers = ["Submission name", "Score"]
        self.filename = "scores.csv"
        if autoload:
            self.load()

    def add_record(self, name, score):
        self.scoreboard.append([name, score])
        self.sort()
        self.flush()
        return self.top()

    def sort(self):
        self.scoreboard.sort(key=lambda x: x[1], reverse=True)
    
    def top(self, count: int = 10):
        return self.scoreboard[:count]
    
    def flush(self):
        with open(self.filename, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(self.headers)
            csvwriter.writerows(self.scoreboard)

    def load(self):
        if not os.path.isfile(self.filename):
            return -1
        
        rows = []
        
        with open(self.filename, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)
            for row in csvreader:
                rows.append(row)

        self.scoreboard = rows
        
    
    
app = Flask(__name__)




@app.route("/scoreboard", methods=['GET'])
def scoreboard_func():
    k1 = request.args.get("len")
    k = (10 if not k1 else k1)
    records = scoreboard.top(count=k)
    return_json = {
        "records": []
    }
    for record in records:
        return_json['records'].append({
            "name": record[0],
            "score": record[1]
        })
    return return_json


@app.route("/add_record", methods=['POST'])
def add_func():
    data = request.get_json(force=True)
    print(data)
    try:
        data['score'] = int(data['score'])
    except Exception:
        return "KAL"
    scoreboard.add_record(data['name'], data['score'])
    return "KHATAM"



if __name__ == "__main__":
    scoreboard = ScoreBoard()
    app.run(host="0.0.0.0")





