from flask import Flask, render_template, session,url_for,redirect,request
from flask_sqlalchemy import SQLAlchemy
import bcrypt  
import pickle
import csv
app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)
app.secret_key = 'secret_key'

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))

    def __init__(self,email,password,name):
        self.name = name
        self.email = email
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def check_password(self,password):
        return bcrypt.checkpw(password.encode('utf-8'),self.password.encode('utf-8'))

with app.app_context():
    db.create_all()



if(__name__=="__main__"):
    app.run(debug=True)
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/register',methods=['GET','POST'])
def register():
    if request.method=='POST':
        #HANDLE THE REQUEST
        name=request.form['name']
        email = request.form['email']
        password = request.form['password']

        new_user = User(name=name,email=email,password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect('/login')
    return render_template('register.html')
@app.route('/login',methods=['GET','POST'])
def login():
    '''if request.method=='POST':
        #HANDLE THE REQUEST
    
        email = request.form['email']
        password = request.form['password']

        #user = User.query.filter_by(email=email).first()
        
        if user and user.check_password(password):
            session['email'] = user.email
            return redirect('/index.html')
        else:
            return render_template('login.html',error='Incorrect password or email')'''
    return render_template('index.html')


@app.route('/submit_form', methods = ['GET','POST'] )
def submit():
    if request.method == "POST":
        try:
            data = request.form.to_dict()
            write_data_csv(data)
            message = 'Form Submitted, Thank you for reaching out!'
            return render_template('thankyou.html',message=message)
        except:
            message = "DID NOT SAVE DATA TO DATABASE."
            return render_template('thankyou.html',message=message)
    else:
        message = "FORM NOT SUBMITTED"
        return render_template('thankyou.html',message=message)

headers = ("transaction_id", "transaction_amount", "customer_id")
data = [(79, 42.1362, 1514),
 (977, 29.0103, 1642),
 (30, 22.2563, 1315),
 (118, 71.27, 1284),
 (49, 43.4614, 1284),
 (741, 16.3305, 1284),
 (453, 37.3568, 1963),
 (101, 39.3557, 1355),
 (658, 20.7447, 1355),
 (959, 98.4384, 1355),
 (686, 59.9888, 1967),
 (780, 57.6133, 1963),
 (151, 30.7449, 1624),
 (964, 70.4956, 1086),
 (550, 52.9478, 1348),
 (994, 16.7989, 1250),
 (348, 60.8966, 1612),
 (481, 59.9973, 1746),
 (778, 12.8196, 1086),
 (860, 43.6371, 1086),
 (983, 64.4469, 1642)]



@app.route('/<string:page_name>')
def page(page_name='/'):
    try:
        return render_template(page_name, headers=headers,data=data)
    except:
        return redirect('/')



def write_data_csv(data):
    email = data['email']
    subject = data['subject']
    message = data["message"]
    with open('db.csv', 'a', newline='') as csvfile:
        db_writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        db_writer.writerow([email,subject,message])