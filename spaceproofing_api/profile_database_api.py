# Import necessary modules and packages
from flask import Flask, request
from flask_restx import Api, Resource, Namespace, fields
from flask_sqlalchemy import SQLAlchemy
import csv
import re
import json

# Initialize Flask application
app = Flask(__name__)

# Configure SQLite database connection
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///db.sqllite"

# Initialize Flask-RestX API
api = Api(app)

# Initialize SQLAlchemy database
db = SQLAlchemy(app)

# Define namespace for API endpoints
ns = Namespace("Spaceproofing Profiles")
api.add_namespace(ns)


# Define SQLAlchemy model for Profile data
class Profile(db.Model):
    id = db.Column(db.Integer, primary_key=True, unique=True)
    profile_id = db.Column(db.String(15), nullable=True)
    name = db.Column(db.String(15), nullable=False)
    control = db.Column(db.String(4), nullable=False)
    date = db.Column(db.String(10), nullable=False)
    cHeight = db.Column(db.Float(precision=3), nullable=False)
    hitchHeightR = db.Column(db.Float(precision=3), nullable=False)
    hitchHeightL = db.Column(db.Float(precision=3), nullable=False)
    cSpan = db.Column(db.Float(precision=3), nullable=False)
    cSpanRoadL = db.Column(db.Float(precision=3), nullable=False)
    carriageWayWidth = db.Column(db.Float(precision=3), nullable=False)
    crossFall = db.Column(db.Float(precision=2), nullable=False)
    xpWidenL = db.Column(db.Float(precision=3), nullable=False)
    xpWidenR = db.Column(db.Float(precision=3), nullable=False)
    tobL = db.Column(db.Float(precision=3), nullable=False)
    tobR = db.Column(db.Float(precision=3), nullable=False)
    lepL = db.Column(db.Float(precision=3), nullable=False)
    lepR = db.Column(db.Float(precision=3), nullable=False)
    tobLCTR = db.Column(db.Float(precision=3), nullable=False)
    tobRCTR = db.Column(db.Float(precision=3), nullable=False)
    wallLCTR = db.Column(db.Float(precision=3), nullable=False)
    wallRCTR = db.Column(db.Float(precision=3), nullable=False)
    dpk = db.Column(db.String(5), nullable=True)

    # Constructor method for Profile class
    def __init__(
        self,
        profile_id,
        name,
        control,
        date,
        cHeight,
        hitchHeightR,
        hitchHeightL,
        cSpan,
        cSpanRoadL,
        carriageWayWidth,
        crossFall,
        xpWidenL,
        xpWidenR,
        tobL,
        tobR,
        lepL,
        lepR,
        tobLCTR,
        tobRCTR,
        wallLCTR,
        wallRCTR,
        dpk,
    ):
        self.profile_id = profile_id
        self.name = name
        self.control = control
        self.date = date
        # Ensure cHeight is less than 30m
        if cHeight > 30:
            raise ValueError("Tunnel Height cannot be greater than 30m")
        self.cHeight = cHeight
        self.hitchHeightR = hitchHeightR
        self.hitchHeightL = hitchHeightL
        if cSpan > 40:
            raise ValueError("Tunnel Span cannot be greater than 40m")
        self.cSpan = cSpan
        self.cSpanRoadL = cSpanRoadL
        self.carriageWayWidth = carriageWayWidth
        self.crossFall = crossFall
        self.xpWidenL = xpWidenL
        self.xpWidenR = xpWidenR
        self.tobL = tobL
        self.tobR = tobR
        self.lepL = lepL
        self.lepR = lepR
        self.tobLCTR = tobLCTR
        self.tobRCTR = tobRCTR
        self.wallLCTR = wallLCTR
        self.wallRCTR = wallRCTR
        self.dpk = dpk

    def serialize(self):
        return {
            "id": self.id,
            "profile_id": self.profile_id,
            "name": self.name,
            "control": self.control,
            "date": self.date,
            "cHeight": self.cHeight,
            "hitchHeightR": self.hitchHeightR,
            "hitchHeightL": self.hitchHeightL,
            "cSpan": self.cSpan,
            "cSpanRoadL": self.cSpanRoadL,
            "carriageWayWidth": self.carriageWayWidth,
            "crossFall": self.crossFall,
            "xpWidenL": self.xpWidenL,
            "xpWidenR": self.xpWidenR,
            "tobL": self.tobL,
            "tobR": self.tobR,
            "lepL": self.lepL,
            "lepR": self.lepR,
            "tobLCTR": self.tobLCTR,
            "tobRCTR": self.tobRCTR,
            "wallLCTR": self.wallLCTR,
            "wallRCTR": self.wallRCTR,
            "dpk": self.dpk,
        }

    # Static method to validate control format
    @staticmethod
    def validate_control(value):
        pattern = re.compile(r"^M[a-zA-Z\d]\d{2}$")
        return bool(pattern.match(value))

    # Static method to validate date format
    @staticmethod
    def validate_date(value):
        pattern = re.compile(r"^\d{2}/\d{2}/\d{4}$")
        return bool(pattern.match(value))


# Function to load data from CSV file into the database
def load_data_from_csv():
    with open("profile_summary_nm.csv", "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            profile = Profile(
                profile_id=row["profile_id"],
                name=row["name"],
                control=row["control"],
                date=row["date"],
                cHeight=float(row["cHeight"]),
                hitchHeightR=float(row["hitchHeightR"]),
                hitchHeightL=float(row["hitchHeightL"]),
                cSpan=float(row["cSpan"]),
                cSpanRoadL=float(row["cSpanRoadL"]),
                carriageWayWidth=float(row["carriageWayWidth"]),
                crossFall=float(row["crossFall"]),
                xpWidenL=float(row["xpWidenL"]),
                xpWidenR=float(row["xpWidenR"]),
                tobL=float(row["tobL"]),
                tobR=float(row["tobR"]),
                lepL=float(row["lepL"]),
                lepR=float(row["lepR"]),
                tobLCTR=float(row["tobLCTR"]),
                tobRCTR=float(row["tobRCTR"]),
                wallLCTR=float(row["wallLCTR"]),
                wallRCTR=float(row["wallRCTR"]),
                dpk=row["dpk"],
            )
            db.session.add(profile)
        db.session.commit()


# Define data model for Profile resource
profile_model = api.model(
    "Profile",
    {
        "id": fields.Integer(description="Profile DB ID"),
        "profile_id": fields.String(description="Profile Long Name"),
        "name": fields.String(description="Profile Short Name"),
        "control": fields.String(description="CTRL Name"),
        "date": fields.String(description="Date Created/Updated"),
        "cHeight": fields.Float(description="Tunnel Height (m)"),
        "hitchHeightR": fields.Float(description="Hitch Height Right (m)"),
        "hitchHeightL": fields.Float(description="Hitch Height Left (m)"),
        "cSpan": fields.Float(description="Tunnel Span (m)"),
        "cSpanRoadL": fields.Float(description="Tunnel Span Road Left (m)"),
        "carriageWayWidth": fields.Float(description="Carriage Way Width (m)"),
        "crossFall": fields.Float(description="Cross Fall (m)"),
        "xpWidenL": fields.Float(description="Xp Widen Left (m)"),
        "xpWidenR": fields.Float(description="Xp Widen Right (m)"),
        "tobL": fields.Float(description="TOB Left (m)"),
        "tobR": fields.Float(description="TOB Right (m)"),
        "lepL": fields.Float(description="LEP Left (m)"),
        "lepR": fields.Float(description="LEP Right (m)"),
        "tobLCTR": fields.Float(description="TOB Left CTR (m)"),
        "tobRCTR": fields.Float(description="TOB Right CTR (m)"),
        "wallLCTR": fields.Float(description="Wall Left CTR (m)"),
        "wallRCTR": fields.Float(description="Wall Right CTR (m)"),
        "dpk": fields.String(description="DPK"),
    },
)


# Define API endpoints for managing profiles
@ns.route("/profile")
class ProfilesListAPI(Resource):
    # Endpoint to retrieve all profiles
    @ns.marshal_list_with(profile_model)
    def get(self):
        return Profile.query.all()


@ns.route("/profiles/<int:id>")
class ProfileAPI(Resource):
    # Endpoint to retrieve a specific profile by ID
    @ns.marshal_with(profile_model)
    def get(self, id):
        profile = Profile.query.get(id)
        return profile

    # Endpoint to update a specific profile by ID
    @ns.expect(profile_model)
    @ns.marshal_with(profile_model)
    def put(self, id):
        profile = Profile.query.get(id)

        if not profile:
            return {"message": "Profile not found"}, 404

        # Update profile attributes
        profile.profile_id = ns.payload["profile_id"]
        profile.name = ns.payload["name"]
        profile.date = ns.payload["date"]
        profile.cHeight = ns.payload["cHeight"]
        profile.hitchHeightR = ns.payload["hitchHeightR"]
        profile.hitchHeightL = ns.payload["hitchHeightL"]
        profile.cSpan = ns.payload["cSpan"]
        profile.carriageWayWidth = ns.payload["carriageWayWidth"]

        db.session.commit()
        return profile

    # Endpoint to delete a specific profile by ID
    @ns.expect(profile_model)
    def delete(self, id):
        profile = Profile.query.get(id)
        db.session.delete(profile)
        db.session.commit()
        return {}, 204


@ns.route("/profiles")
class ProfileList(Resource):
    # Endpoint to create a new profile
    @ns.expect(profile_model)
    @ns.marshal_with(profile_model, code=201)
    def post(self):
        data = request.json

        # Extract profile data from request
        profile_id = data.get("profile_id")
        name = data.get("name")
        control = data.get("control")
        date = data.get("date")
        cHeight = data.get("cHeight")
        hitchHeightR = data.get("hitchHeightR")
        hitchHeightL = data.get("hitchHeightL")
        cSpan = data.get("cSpan")
        carriageWayWidth = data.get("carriageWayWidth")

        # Validate control format
        if not Profile.validate_control(control):
            return {
                "message": "Invalid control format. Control format must be like 'M[A-Z][0-9]{2}'"
            }, 400

        # Validate date format
        if not Profile.validate_date(date):
            return {
                "message": "Invalid date format. Date format must be like 'DD/MM/YYYY'"
            }, 400

        # Create a new profile object
        new_profile = Profile(
            profile_id=profile_id,
            name=name,
            control=control,
            date=date,
            cHeight=cHeight,
            hitchHeightR=hitchHeightR,
            hitchHeightL=hitchHeightL,
            cSpan=cSpan,
            carriageWayWidth=carriageWayWidth,
        )

        # Add new profile to the database
        db.session.add(new_profile)
        db.session.commit()

        return new_profile, 201


def save_profiles_to_json():
    # Fetch all profiles from the database
    all_profiles = Profile.query.all()

    # Serialize profiles to JSON format
    profiles_json = [profile.serialize() for profile in all_profiles]

    # Write JSON data to a file
    with open("profiles.json", "w") as json_file:
        json.dump(profiles_json, json_file, indent=4)


# Create all database tables within the application context
# Comment out once database is created
# with app.app_context():
# db.create_all()
# Load data from CSV into the database
# load_data_from_csv()
# Call the function to save profiles to JSON
# save_profiles_to_json()


# Run the Flask application
if __name__ == "__main__":
    app.run(debug=True)
