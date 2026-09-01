from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# =========================================================
# LOAD ML MODELS
# =========================================================

score_model_path = os.path.join(BASE_DIR, "ipl_model.pkl")
speed_model_path = os.path.join(BASE_DIR, "speed_model.pkl")
strike_rate_model_path = os.path.join(BASE_DIR, "strike_rate_model.pkl")
win_model_path = os.path.join(BASE_DIR, "win_model.pkl")

model = joblib.load(score_model_path)
speed_model = joblib.load(speed_model_path)
strike_rate_model = joblib.load(strike_rate_model_path)
win_model = joblib.load(win_model_path)


# =========================================================
# IPL PLAYERS
# =========================================================

IPL_PLAYERS = {

    "CSK": [
        ("Ruturaj Gaikwad", "Batter"),
        ("MS Dhoni", "Wicket Keeper"),
        ("Ravindra Jadeja", "All-Rounder"),
        ("Shivam Dube", "All-Rounder"),
        ("Dewald Brevis", "Batter"),
        ("Ayush Mhatre", "Batter"),
        ("Sanju Samson", "Wicket Keeper"),
        ("Sarfaraz Khan", "Batter"),
        ("Matthew Short", "All-Rounder"),
        ("Prashant Veer", "All-Rounder"),
        ("Akeal Hosein", "All-Rounder"),
        ("Noor Ahmad", "Bowler"),
        ("Rahul Chahar", "Bowler"),
        ("Khaleel Ahmed", "Bowler"),
        ("Mukesh Choudhary", "Bowler"),
        ("Gurjapneet Singh", "Bowler"),
        ("Anshul Kamboj", "Bowler"),
        ("Spencer Johnson", "Bowler"),
        ("Matt Henry", "Bowler"),
        ("Nathan Ellis", "Bowler"),
        ("Akash Madhwal", "Bowler"),
        ("Shreyas Gopal", "Bowler"),
        ("Kartik Sharma", "Wicket Keeper"),
        ("Urvil Patel", "Wicket Keeper"),
        ("Vansh Bedi", "Batter")
    ],

    "MI": [
        ("Rohit Sharma", "Batter"),
        ("Suryakumar Yadav", "Batter"),
        ("Tilak Varma", "Batter"),
        ("Hardik Pandya", "All-Rounder"),
        ("Jasprit Bumrah", "Bowler"),
        ("Trent Boult", "Bowler"),
        ("Deepak Chahar", "Bowler"),
        ("Ryan Rickelton", "Wicket Keeper"),
        ("Robin Minz", "Wicket Keeper"),
        ("Will Jacks", "All-Rounder"),
        ("Mitchell Santner", "All-Rounder"),
        ("Naman Dhir", "All-Rounder"),
        ("Shams Mulani", "All-Rounder"),
        ("Raj Angad Bawa", "All-Rounder"),
        ("Sherfane Rutherford", "All-Rounder"),
        ("Mayank Markande", "Bowler"),
        ("Ashwani Kumar", "Bowler"),
        ("Arjun Tendulkar", "Bowler"),
        ("Vignesh Puthur", "Bowler"),
        ("Allah Ghazanfar", "Bowler"),
        ("Anmolpreet Singh", "Batter"),
        ("Danish Malewar", "Batter"),
        ("Mayank Rawat", "Batter"),
        ("Keshav Mahajan", "Bowler"),
        ("Corbin Bosch", "All-Rounder")
    ],

    "RCB": [
        ("Virat Kohli", "Batter"),
        ("Rajat Patidar", "Batter"),
        ("Phil Salt", "Wicket Keeper"),
        ("Jitesh Sharma", "Wicket Keeper"),
        ("Devdutt Padikkal", "Batter"),
        ("Krunal Pandya", "All-Rounder"),
        ("Tim David", "All-Rounder"),
        ("Romario Shepherd", "All-Rounder"),
        ("Jacob Bethell", "All-Rounder"),
        ("Swapnil Singh", "All-Rounder"),
        ("Venkatesh Iyer", "All-Rounder"),
        ("Josh Hazlewood", "Bowler"),
        ("Bhuvneshwar Kumar", "Bowler"),
        ("Yash Dayal", "Bowler"),
        ("Suyash Sharma", "Bowler"),
        ("Rasikh Salam", "Bowler"),
        ("Nuwan Thushara", "Bowler"),
        ("Abhinandan Singh", "Bowler"),
        ("Mohit Rathee", "Bowler"),
        ("Jacob Duffy", "Bowler"),
        ("Saurav Chauhan", "Batter"),
        ("Manoj Bhandage", "All-Rounder"),
        ("Swastik Chikara", "Batter"),
        ("Vihaan Malhotra", "Batter"),
        ("Jordan Cox", "Wicket Keeper")
    ],

    "KKR": [
        ("Ajinkya Rahane", "Batter"),
        ("Rinku Singh", "Batter"),
        ("Angkrish Raghuvanshi", "Batter"),
        ("Manish Pandey", "Batter"),
        ("Rahul Tripathi", "Batter"),
        ("Finn Allen", "Wicket Keeper"),
        ("Quinton de Kock", "Wicket Keeper"),
        ("Sunil Narine", "All-Rounder"),
        ("Venkatesh Iyer", "All-Rounder"),
        ("Rovman Powell", "All-Rounder"),
        ("Anukul Roy", "All-Rounder"),
        ("Ramandeep Singh", "All-Rounder"),
        ("Vaibhav Arora", "Bowler"),
        ("Varun Chakravarthy", "Bowler"),
        ("Harshit Rana", "Bowler"),
        ("Mustafizur Rahman", "Bowler"),
        ("Navdeep Saini", "Bowler"),
        ("Prashant Solanki", "Bowler"),
        ("Saurabh Dubey", "Bowler"),
        ("Spencer Johnson", "Bowler"),
        ("Chetan Sakariya", "Bowler"),
        ("Mayank Markande", "Bowler"),
        ("Rachin Ravindra", "All-Rounder"),
        ("Luvnith Sisodia", "Wicket Keeper"),
        ("Vaibhav Arora", "Bowler")
    ],

    "SRH": [
        ("Travis Head", "Batter"),
        ("Abhishek Sharma", "All-Rounder"),
        ("Ishan Kishan", "Wicket Keeper"),
        ("Heinrich Klaasen", "Wicket Keeper"),
        ("Rahul Tripathi", "Batter"),
        ("Aniket Verma", "Batter"),
        ("Sachin Baby", "Batter"),
        ("Nitish Kumar Reddy", "All-Rounder"),
        ("Washington Sundar", "All-Rounder"),
        ("Pat Cummins", "Bowler"),
        ("Harshal Patel", "Bowler"),
        ("Jaydev Unadkat", "Bowler"),
        ("Adam Zampa", "Bowler"),
        ("Eshan Malinga", "Bowler"),
        ("Simarjeet Singh", "Bowler"),
        ("Zeeshan Ansari", "Bowler"),
        ("Brydon Carse", "All-Rounder"),
        ("Kamindu Mendis", "All-Rounder"),
        ("Abhinav Manohar", "Batter"),
        ("Atharva Taide", "Batter"),
        ("Shivam Sharma", "Bowler"),
        ("Smaran Ravichandran", "Batter"),
        ("Jayant Yadav", "All-Rounder"),
        ("Mayank Agarwal", "Batter"),
        ("Umran Malik", "Bowler")
    ],

    "RR": [
        ("Yashasvi Jaiswal", "Batter"),
        ("Sanju Samson", "Wicket Keeper"),
        ("Dhruv Jurel", "Wicket Keeper"),
        ("Vaibhav Sooryavanshi", "Batter"),
        ("Shimron Hetmyer", "Batter"),
        ("Riyan Parag", "All-Rounder"),
        ("Jofra Archer", "Bowler"),
        ("Tushar Deshpande", "Bowler"),
        ("Sandeep Sharma", "Bowler"),
        ("Maheesh Theekshana", "Bowler"),
        ("Wanindu Hasaranga", "All-Rounder"),
        ("Fazalhaq Farooqi", "Bowler"),
        ("Kwena Maphaka", "Bowler"),
        ("Shubham Dubey", "Batter"),
        ("Kunal Rathore", "Wicket Keeper"),
        ("Yudhvir Singh", "All-Rounder"),
        ("Ashok Sharma", "Bowler"),
        ("Kumar Kartikeya", "Bowler"),
        ("Ravichandran Ashwin", "All-Rounder"),
        ("Donovan Ferreira", "Wicket Keeper"),
        ("Nandre Burger", "Bowler"),
        ("Abid Mushtaq", "All-Rounder"),
        ("Akash Singh", "Bowler"),
        ("Theekshana", "Bowler"),
        ("Kunal Rathore", "Wicket Keeper")
    ],

    "DC": [
        ("KL Rahul", "Wicket Keeper"),
        ("Abhishek Porel", "Wicket Keeper"),
        ("Karun Nair", "Batter"),
        ("Nitish Rana", "Batter"),
        ("Sameer Rizvi", "Batter"),
        ("Tristan Stubbs", "Batter"),
        ("Faf du Plessis", "Batter"),
        ("Axar Patel", "All-Rounder"),
        ("Mitchell Marsh", "All-Rounder"),
        ("Ashutosh Sharma", "All-Rounder"),
        ("Vipraj Nigam", "All-Rounder"),
        ("Kuldeep Yadav", "Bowler"),
        ("Mukesh Kumar", "Bowler"),
        ("T Natarajan", "Bowler"),
        ("Mitchell Starc", "Bowler"),
        ("Dushmantha Chameera", "Bowler"),
        ("Mohit Sharma", "Bowler"),
        ("Nandre Burger", "Bowler"),
        ("Ajay Mandal", "All-Rounder"),
        ("Darshan Nalkande", "All-Rounder"),
        ("Jake Fraser-McGurk", "Batter"),
        ("Siddhant Sharma", "Bowler"),
        ("Madhav Tiwari", "All-Rounder"),
        ("Shai Hope", "Wicket Keeper"),
        ("Swastik Chikara", "Batter")
    ],

    "PBKS": [
        ("Shreyas Iyer", "Batter"),
        ("Priyansh Arya", "Batter"),
        ("Nehal Wadhera", "Batter"),
        ("Prabhsimran Singh", "Wicket Keeper"),
        ("Josh Inglis", "Wicket Keeper"),
        ("Shashank Singh", "All-Rounder"),
        ("Marcus Stoinis", "All-Rounder"),
        ("Marco Jansen", "All-Rounder"),
        ("Glenn Maxwell", "All-Rounder"),
        ("Azmatullah Omarzai", "All-Rounder"),
        ("Arshdeep Singh", "Bowler"),
        ("Yuzvendra Chahal", "Bowler"),
        ("Kagiso Rabada", "Bowler"),
        ("Lockie Ferguson", "Bowler"),
        ("Vijaykumar Vyshak", "Bowler"),
        ("Harpreet Brar", "All-Rounder"),
        ("Yash Thakur", "Bowler"),
        ("Xavier Bartlett", "Bowler"),
        ("Musheer Khan", "All-Rounder"),
        ("Suryansh Shedge", "All-Rounder"),
        ("Aaron Hardie", "All-Rounder"),
        ("Vishnu Vinod", "Wicket Keeper"),
        ("Priyanshu Moliya", "Batter"),
        ("Harnoor Singh", "Batter"),
        ("Kuldeep Sen", "Bowler")
    ],

    "GT": [
        ("Shubman Gill", "Batter"),
        ("Sai Sudharsan", "Batter"),
        ("Jos Buttler", "Wicket Keeper"),
        ("Kumar Kushagra", "Wicket Keeper"),
        ("Anuj Rawat", "Wicket Keeper"),
        ("Glenn Phillips", "Batter"),
        ("Sherfane Rutherford", "All-Rounder"),
        ("Washington Sundar", "All-Rounder"),
        ("Rahul Tewatia", "All-Rounder"),
        ("Shahrukh Khan", "All-Rounder"),
        ("Rashid Khan", "Bowler"),
        ("Mohammed Siraj", "Bowler"),
        ("Prasidh Krishna", "Bowler"),
        ("Ishant Sharma", "Bowler"),
        ("Kagiso Rabada", "Bowler"),
        ("Gerald Coetzee", "Bowler"),
        ("Arshad Khan", "All-Rounder"),
        ("R Sai Kishore", "Bowler"),
        ("Manav Suthar", "Bowler"),
        ("Jayant Yadav", "All-Rounder"),
        ("Mahipal Lomror", "All-Rounder"),
        ("Gurnoor Brar", "Bowler"),
        ("Nishant Sindhu", "All-Rounder"),
        ("Shubham Dubey", "Batter"),
        ("Kartik Tyagi", "Bowler")
    ],

    "LSG": [
        ("Rishabh Pant", "Wicket Keeper"),
        ("Aiden Markram", "All-Rounder"),
        ("Mitchell Marsh", "All-Rounder"),
        ("Nicholas Pooran", "Wicket Keeper"),
        ("Ayush Badoni", "Batter"),
        ("David Miller", "Batter"),
        ("Matthew Breetzke", "Batter"),
        ("Abdul Samad", "All-Rounder"),
        ("Shahbaz Ahmed", "All-Rounder"),
        ("Ravi Bishnoi", "Bowler"),
        ("Avesh Khan", "Bowler"),
        ("Mayank Yadav", "Bowler"),
        ("Mohsin Khan", "Bowler"),
        ("Shamar Joseph", "Bowler"),
        ("Akash Deep", "Bowler"),
        ("Digvesh Singh", "Bowler"),
        ("Prince Yadav", "All-Rounder"),
        ("M Siddharth", "Bowler"),
        ("Arshin Kulkarni", "All-Rounder"),
        ("Yuvraj Chaudhary", "All-Rounder"),
        ("Rajvardhan Hangargekar", "Bowler"),
        ("Himmat Singh", "Batter"),
        ("Aryan Juyal", "Wicket Keeper"),
        ("Shah Rukh Khan", "All-Rounder"),
        ("Abhinand Singh", "Bowler")
    ]
}


# =========================================================
# TEAM STRENGTH
# =========================================================

TEAM_STRENGTH = {
    "CSK": 82,
    "MI": 86,
    "RCB": 84,
    "KKR": 83,
    "SRH": 85,
    "RR": 80,
    "DC": 79,
    "PBKS": 81,
    "GT": 84,
    "LSG": 80
}


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():
    return render_template("home.html")


# =========================================================
# TEAMS
# =========================================================

@app.route("/teams")
def teams():
    return render_template("teams.html")


# =========================================================
# PLAYERS
# =========================================================

@app.route("/players/<team>")
def players(team):

    team = team.replace("'''", "")
    team = team.replace('"""', "")
    team = team.replace("`", "")
    team = team.strip().upper()

    players_list = IPL_PLAYERS.get(team, [])

    return render_template(
        "players.html",
        team=team,
        players=players_list
    )


# =========================================================
# START PREDICTION
# =========================================================

@app.route("/start_prediction")
def start_prediction():

    return render_template("predict.html")


# =========================================================
# PREDICTION
# =========================================================

@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "POST":

        batting_team = request.form["batting_team"]
        bowling_team = request.form["bowling_team"]

        batter = request.form.get("batter", "")
        bowler = request.form.get("bowler", "")

        # PITCH TYPE
        pitch_type = request.form.get("pitch_type", "")

        score = float(request.form["score"])
        overs = float(request.form["overs"])
        wickets = int(request.form["wickets"])


        # =================================================
        # RUN RATE
        # =================================================

        if overs > 0:
            run_rate = score / overs
        else:
            run_rate = 0


        # =================================================
        # SCORE MODEL INPUT
        # =================================================

        input_data = pd.DataFrame({

            "BattingTeam": [batting_team],

            "BowlingTeam": [bowling_team],

            "CurrentScore": [score],

            "Overs": [overs],

            "Wickets": [wickets],

            "RunRate": [run_rate]

        })


        # =================================================
        # FINAL SCORE PREDICTION
        # =================================================

        predicted_score = model.predict(input_data)[0]

        predicted_score = max(
            0,
            round(predicted_score)
        )


        # =================================================
        # SCORE RANGE
        # =================================================

        min_score = max(
            0,
            predicted_score - 10
        )

        max_score = predicted_score + 10


        # =================================================
        # BALL SPEED
        # =================================================

        speed_input = pd.DataFrame({

            "bowler": ["Unknown"],

            "bowlingStyle": ["Unknown"],

            "rightArmedBowl": [1],

            "ball": [1],

            "pitchX": [0],

            "pitchY": [0]

        })

        try:

            predicted_speed = speed_model.predict(
                speed_input
            )[0]

            predicted_speed = round(
                float(predicted_speed),
                2
            )

            predicted_speed = max(
                0,
                predicted_speed
            )

        except Exception:

            predicted_speed = 0


        # =================================================
        # STRIKE RATE
        # =================================================

        balls_faced = max(
            1,
            round(overs * 6)
        )

        cumulative_runs = max(
            0,
            score
        )

        strike_input = pd.DataFrame({

            "BallNumber": [balls_faced],

            "CumulativeRuns": [cumulative_runs]

        })

        try:

            predicted_strike_rate = strike_rate_model.predict(
                strike_input
            )[0]

            predicted_strike_rate = round(
                float(predicted_strike_rate),
                2
            )

            predicted_strike_rate = max(
                0,
                predicted_strike_rate
            )

        except Exception:

            predicted_strike_rate = 0


        # =================================================
        # WINNING TEAM - ML MODEL
        # =================================================

        try:

            win_input = pd.DataFrame({

                "BattingTeam": [batting_team],

                "BowlingTeam": [bowling_team],

                "CurrentScore": [score],

                "Overs": [overs],

                "Wickets": [wickets],

                "RunRate": [run_rate],

                "PitchType": [pitch_type]

            })


            # ML prediction
            win_prediction = win_model.predict(
                win_input
            )[0]


            # =================================================
            # CONVERT ML OUTPUT TO TEAM NAME
            # =================================================

            prediction_text = str(
                win_prediction
            ).strip()


            # Model directly returns team name
            if prediction_text in [
                batting_team,
                bowling_team
            ]:

                winning_team = prediction_text


            # Model returns 0 = batting team
            elif prediction_text in ["0", "0.0"]:

                winning_team = batting_team


            # Model returns 1 = bowling team
            elif prediction_text in ["1", "1.0"]:

                winning_team = bowling_team


            else:

                try:

                    numeric_prediction = int(
                        float(win_prediction)
                    )

                    if numeric_prediction == 0:

                        winning_team = batting_team

                    else:

                        winning_team = bowling_team

                except Exception:

                    winning_team = batting_team


            # =================================================
            # WINNING PROBABILITY
            # =================================================

            if hasattr(
                win_model,
                "predict_proba"
            ):

                probabilities = win_model.predict_proba(
                    win_input
                )[0]

                classes = list(
                    win_model.classes_
                )


                probability_map = dict(
                    zip(
                        [str(c) for c in classes],
                        probabilities
                    )
                )


                predicted_class = str(
                    win_prediction
                )


                if predicted_class in probability_map:

                    win_prob = (
                        probability_map[
                            predicted_class
                        ] * 100
                    )

                else:

                    win_prob = 50.0


            else:

                win_prob = 50.0


        except Exception as e:

            print(
                "Winning ML prediction error:",
                e
            )


            # =================================================
            # FALLBACK
            # =================================================

            batting_strength = TEAM_STRENGTH.get(
                batting_team,
                80
            )

            bowling_strength = TEAM_STRENGTH.get(
                bowling_team,
                80
            )

            total_strength = (
                batting_strength +
                bowling_strength
            )

            batting_probability = (
                batting_strength /
                total_strength
            ) * 100


            if predicted_score >= 180:

                batting_probability += 12

            elif predicted_score >= 160:

                batting_probability += 7

            elif predicted_score >= 140:

                batting_probability += 3

            elif predicted_score < 120:

                batting_probability -= 10

            elif predicted_score < 140:

                batting_probability -= 5


            if wickets >= 8:

                batting_probability -= 20

            elif wickets >= 6:

                batting_probability -= 12

            elif wickets >= 4:

                batting_probability -= 5


            if run_rate >= 10:

                batting_probability += 6

            elif run_rate >= 8:

                batting_probability += 3

            elif run_rate < 6:

                batting_probability -= 5


            batting_probability = min(
                90,
                max(
                    10,
                    batting_probability
                )
            )


            bowling_probability = (
                100 -
                batting_probability
            )


            if batting_probability >= bowling_probability:

                winning_team = batting_team

                win_prob = batting_probability

            else:

                winning_team = bowling_team

                win_prob = bowling_probability


        win_prob = round(
            float(win_prob),
            2
        )


        # =================================================
        # RESULT
        # =================================================

        return render_template(

            "result.html",

            batting_team=batting_team,

            bowling_team=bowling_team,

            batter=batter,

            bowler=bowler,

            pitch_type=pitch_type,

            score=score,

            overs=overs,

            wickets=wickets,

            run_rate=round(
                run_rate,
                2
            ),

            prediction=predicted_score,

            min_score=min_score,

            max_score=max_score,

            winning_team=winning_team,

            win_prob=win_prob,

            predicted_speed=predicted_speed,

            predicted_strike_rate=predicted_strike_rate

        )


    return render_template("predict.html")


# =========================================================
# START APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )