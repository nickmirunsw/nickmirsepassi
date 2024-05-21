#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
COMP9321 24T1 Assignment 2
Data publication as a RESTful service API

Getting Started
---------------

1. You MUST rename this file according to your zID, e.g., z1234567.py.

2. To ensure your submission can be marked correctly, you're strongly encouraged
   to create a new virtual environment for this assignment.  Please see the
   instructions in the assignment 1 specification to create and activate a
   virtual environment.

3. Once you have activated your virtual environment, you need to install the
   following, required packages:

   pip install python-dotenv==1.0.1
   pip install google-generativeai==0.4.1

   You may also use any of the packages we've used in the weekly labs.
   The most likely ones you'll want to install are:

   pip install flask==3.0.2
   pip install flask_restx==1.3.0
   pip install requests==2.31.0

4. Create a file called `.env` in the same directory as this file.  This file
   will contain the Google API key you generatea in the next step.

5. Go to the following page, click on the link to "Get an API key", and follow
   the instructions to generate an API key:

   https://ai.google.dev/tutorials/python_quickstart

6. Add the following line to your `.env` file, replacing `your-api-key` with
   the API key you generated, and save the file:

   GOOGLE_API_KEY=your-api-key

7. You can now start implementing your solution. You are free to edit this file how you like, but keep it readable
   such that a marker can read and understand your code if necessary for partial marks.

Submission
----------

You need to submit this Python file and a `requirements.txt` file.

The `requirements.txt` file should list all the Python packages your code relies
on, and their versions.  You can generate this file by running the following
command while your virtual environment is active:

pip freeze > requirements.txt

You can submit the two files using the following command when connected to CSE,
and assuming the files are in the current directory (remember to replace `zid`
with your actual zID, i.e. the name of this file after renaming it):

give cs9321 assign2 zid.py requirements.txt

You can also submit through WebCMS3, using the tab at the top of the assignment
page.

"""

# You can import more modules from the standard library here if you need them
# (which you will, e.g. sqlite3).

# NOTE ASSIGNMENT 2 SOLUTION ====================================================================
# Nima Mirsepassi
# Studnet ID: z5437291
# Date: 2021-05-30
# NOTE remove the comment on the first run to create the db


# Import Third-party libraries
from dotenv import load_dotenv
import google.generativeai as genai
from flask import Flask
from flask import request
from flask import send_file
from flask_restx import Api
from flask_restx import Resource
from flask_restx import fields
from flask_restx import reqparse
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func


# import standard libraries
import os
from pathlib import Path
from datetime import datetime
import requests
import json
import io
import random
import pytz


studentid = Path(__file__).stem
print(studentid)

db_file = f"{studentid}.db"
txt_file = f"{studentid}.txt"

load_dotenv()
api_key = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=api_key)
gemini = genai.GenerativeModel("gemini-pro")


app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{db_file}.sqlite"
api = Api(
    app,
    default="Transport",
    title="Transport API",
    description="A smart API for the Deutsche Bahn",
)
db = SQLAlchemy(app)

# question 5 - Define the model for updating a stop
update_stop_model = api.model(
    "UpdateStop",
    {
        "name": fields.String(
            description="Name of the stop", required=False, default=""
        ),
        "latitude": fields.Float(description="Latitude of the stop", required=False),
        "longitude": fields.Float(description="Longitude of the stop", required=False),
        "next_departure": fields.String(
            description="Next departure from the stop", required=False, default=""
        ),
        "last_updated": fields.String(
            description="Last updated timestamp", required=False, default=""
        ),
    },
)


# Define Sydney's time zone
sydney_tz = pytz.timezone("Australia/Sydney")

# Get the current time in UTC
utc_now = datetime.utcnow()

# Convert UTC time to Sydney's time zone
sydney_now = utc_now.replace(tzinfo=pytz.utc).astimezone(sydney_tz)


@api.route("/stops/<string:query>")
class StopsQuery(Resource):
    """
    Endpoint to query stops based on a string query.
    """

    @api.response(404, "Not found")
    @api.response(400, "Validation Error")
    @api.response(503, "Unable to Respond")
    @api.response(201, "Created")
    @api.response(200, "Successful")
    def put(self, query):
        """
        PUT method to retrieve stops based on a string query.

        Args:
            query (str): The search query.

        Returns:
            JSON: Details of the queried stops.
        """
        try:
            # Construct the URL for querying stops based on the provided query
            url = f"https://v6.db.transport.rest/locations?poi=false&addresses=false&query={query}"
            response = requests.get(url, verify=False)

            # Check if the response is successful
            if response.ok:
                # Extract data from the response and limit to 5 results with stop_type = stop
                data = response.json()
                response_data = []
                stop_count = 0
                for item in data:
                    if item.get("type") == "stop":  # Check if it's a stop
                        stop_count += 1
                        if stop_count > 5:  # Limit to 5 stops
                            break
                        # Save or update each stop in the database
                        created_new_stop = save_update_stop_to_db(item)
                        # Retrieve the stop details from the database
                        database_row = Stops.query.filter_by(
                            stop_id=item.get("id")
                        ).first()
                        if database_row:
                            # Construct stop links
                            stop_links = {
                                "self": {
                                    "href": f"http://{request.host}/stops/{database_row.stop_id}"
                                }
                            }
                            # Append stop details to response data
                            response_data.append(
                                {
                                    "id": database_row.stop_id,
                                    "last_updated": database_row.last_updated,
                                    "link": stop_links,
                                }
                            )

                # Sort response_data based on id
                response_data.sort(key=lambda x: x["id"])

                return response_data, 201 if created_new_stop else 200
            else:
                return {
                    "message": "Failed to get data from Deutsche Bahn API",
                    "status_code": response.status_code,
                }, response.status_code

        except Exception as e:
            print(f"An error occurred: {e}")
            return {"message": "Unable to respond, Try again!"}, 503


def save_update_stop_to_db(item):
    """
    Function to save or update a stop in the database.

    Args:
        item (dict): Details of the stop to be saved or updated.

    Returns:
        bool: True if a new stop is created, False if an existing stop is updated.
    """
    # Extract latitude and longitude from the item
    latitude = item.get("location", {}).get("latitude")
    longitude = item.get("location", {}).get("longitude")

    # Check if the stop already exists in the database
    existing_stop = Stops.query.filter_by(stop_id=item.get("id")).first()

    if existing_stop:
        # Update existing stop details
        existing_stop.stop_type = item.get("type")
        existing_stop.name = item.get("name")
        existing_stop.latitude = latitude
        existing_stop.longitude = longitude
        existing_stop.products = json.dumps(item.get("products"))
        existing_stop.href = f"http://{request.host}/stops/{item.get('id')}"
        existing_stop.last_updated = sydney_now.strftime(
            "%Y-%m-%d-%H:%M:%S"
        )  # Format datetime
        db.session.commit()
        return False
    else:
        # Create a new stop entry
        new_stop = Stops(
            stop_type=item.get("type"),
            stop_id=item.get("id"),
            name=item.get("name"),
            latitude=latitude,
            longitude=longitude,
            products=json.dumps(item.get("products")),
            href=f"http://{request.host}/stops/{item.get('id')}",
            last_updated=sydney_now.strftime("%Y-%m-%d-%H:%M:%S"),
        )
        db.session.add(new_stop)
        db.session.commit()
        return True


stop_parser = reqparse.RequestParser()
stop_parser.add_argument(
    "include",
    type=str,
    help="Attributes to include in response separated by comma",
)


@api.route("/stops/<int:stop_id>")
class StopsDetail(Resource):
    """
    Endpoint to retrieve details of a specific stop by its ID.
    """

    @api.response(404, "Not found")
    @api.response(503, "Unable to Respond")
    @api.response(400, "Validation Error")
    @api.response(200, "Successful")
    @api.expect(stop_parser)
    def get(self, stop_id):
        """
        GET method to retrieve details of a specific stop by its ID.

        Args:
            stop_id (int): The ID of the stop to retrieve details for.

        Returns:
            JSON: Details of the stop including specified attributes.
        """
        try:
            # Retrieve stop details from the database
            stop = Stops.query.filter_by(stop_id=stop_id).first()

            # If stop not found, return 404
            if not stop:
                return {"message": "Stop not found"}, 404

            # Fetch departure information from Deutsche Bahn API
            url = (
                f"https://v6.db.transport.rest/stops/{stop_id}/departures?&duration=120"
            )
            response = requests.get(url, verify=False)

            # If API request is successful, update next departure information
            if response.ok:
                departure_info = response.json().get("departures")
                next_departure = None
                for departure in departure_info:
                    if departure.get("platform") and departure.get("direction"):
                        next_departure = departure
                        break

                # If next departure not found, set next_departure to "Next Departure Not Available"
                if not next_departure:
                    stop.next_departure = "Next Departure Not Available"
                else:
                    stop.next_departure = f"Platform {next_departure.get('platform')} towards {next_departure.get('direction')}"
                db.session.commit()

            else:
                # If API request fails, return error message
                return {
                    "message": "Failed to fetch departures from Deutsche Bahn API",
                    "status_code": response.status_code,
                }, 503

            # Prepare response data
            response_data = {
                "stop_id": stop.stop_id,
                "last_updated": str(stop.last_updated),
                "name": stop.name,
                "latitude": stop.latitude,
                "longitude": stop.longitude,
                "next_departure": stop.next_departure,
            }

            # Check if "include" query parameter is present
            if "include" in request.args:
                # Ensure that "_links" and "stop_id" are not included in the include query parameter
                included_values = [
                    attr.strip() for attr in request.args["include"].split(",")
                ]
                if "_links" in included_values or "stop_id" in included_values:
                    return {
                        "message": "_links and stop_id cannot be included in the 'include' query parameter"
                    }, 400

                # Check for invalid attributes
                invalid_attributes = [
                    attr for attr in included_values if attr not in response_data
                ]
                if invalid_attributes:
                    return {
                        "message": f"{', '.join(invalid_attributes)} is an invalid attribute"
                    }, 400

                response_data = {
                    key: response_data[key]
                    for key in included_values
                    if key in response_data
                }

            response_data["stop_id"] = stop.stop_id

            # Ensure _links are always included
            response_data["_links"] = {
                "self": {"href": f"http://{request.host}/stops/{stop_id}"}
            }

            # Find previous and next stops
            prev_stop = (
                Stops.query.filter(Stops.stop_id < stop_id)
                .order_by(Stops.stop_id.desc())
                .first()
            )
            next_stop = (
                Stops.query.filter(Stops.stop_id > stop_id)
                .order_by(Stops.stop_id)
                .first()
            )

            # Add previous and next links if available
            if prev_stop:
                response_data["_links"]["prev"] = {
                    "href": f"http://{request.host}/stops/{prev_stop.stop_id}"
                }
            if next_stop:
                response_data["_links"]["next"] = {
                    "href": f"http://{request.host}/stops/{next_stop.stop_id}"
                }

            return response_data, 200

        except Exception as e:
            print(f"An error occurred: {e}")
            return {"message": "Unable to respond, Try again!"}, 503

    @api.response(404, "Not found")
    @api.response(400, "Validation Error")
    @api.response(200, "Successful")
    def delete(self, stop_id):
        """
        DELETE method to remove a stop from the database by its ID.

        Args:
            stop_id (int): The ID of the stop to remove.

        Returns:
            JSON: Message confirming the removal of the stop.
        """
        try:
            # Check if the stop exists
            stop_to_remove = Stops.query.filter_by(stop_id=stop_id).first()
            if not stop_to_remove:
                return {
                    "message": f"The stop with ID {stop_id} was not found in the database",
                    "stop_id": stop_id,
                }, 404

            # Delete the stop from the database
            db.session.delete(stop_to_remove)
            db.session.commit()

            # Return success message
            return (
                {
                    "message": f"The stop with ID {stop_id} was removed from the database",
                    "stop_id": stop_id,
                },
                200,
            )

        except Exception as e:
            print(f"An error occurred: {e}")
            db.session.rollback()
            return {"message": "An error occurred while deleting the stop"}, 400

    @api.expect(update_stop_model)
    @api.response(404, "Not found")
    @api.response(400, "Validation Error")
    @api.response(200, "Successful")
    def put(self, stop_id):
        """
        PUT method to update details of a specific stop by its ID.

        Args:
            stop_id (int): The ID of the stop to update.

        Returns:
            JSON: Updated details of the stop.
        """
        try:
            data = request.json

            # Check if prohibited fields are present
            prohibited_fields = ["_links", "stop_id"]
            if any(field in prohibited_fields for field in data):
                return {
                    "message": f"Prohibited fields: {', '.join(prohibited_fields)}"
                }, 400

            # Retrieve the stop from the database
            stop = Stops.query.filter_by(stop_id=stop_id).first()

            # If stop not found, return 404
            if not stop:
                return {"message": "Stop not found"}, 404

            # If no data provided, return 400
            if not data:
                return {"message": "No data provided"}, 400

            # Check for empty fields
            invalid_fields = [
                field
                for field in data
                if field
                not in [
                    "name",
                    "latitude",
                    "longitude",
                    "next_departure",
                    "last_updated",
                ]
            ]
            if invalid_fields:
                return {"message": f"Invalid fields: {', '.join(invalid_fields)}"}, 400

            # Validate and update stop attributes
            for field, value in data.items():
                if field == "last_updated":
                    try:
                        datetime.strptime(value, "%Y-%m-%d-%H:%M:%S")
                    except ValueError:
                        return {
                            "message": "Invalid last_updated format. Use YYYY-MM-DD-HH:MM:SS"
                        }, 400
                    setattr(stop, field, value)
                elif field == "latitude":
                    try:
                        latitude = float(value)
                    except ValueError:
                        return {"message": "Latitude must be a float"}, 400
                    setattr(stop, field, latitude)
                elif field == "longitude":
                    try:
                        longitude = float(value)
                    except ValueError:
                        return {"message": "Longitude must be a float"}, 400
                    setattr(stop, field, longitude)
                elif field == "name" or field == "next_departure":
                    if not isinstance(value, str) or not value.strip():
                        return {
                            "message": f"{field.capitalize()} must be a non-empty string"
                        }, 400
                    setattr(stop, field, value)

            # If last_updated not provided, update it with current timestamp
            if "last_updated" not in data:
                stop.last_updated = sydney_now.strftime("%Y-%m-%d-%H:%M:%S")

            # Commit changes to the database
            db.session.commit()

            # Prepare response data
            response_data = {
                "stop_id": stop.stop_id,
                "last_updated": stop.last_updated,
                "_links": {"self": {"href": f"http://{request.host}/stops/{stop_id}"}},
            }

            return response_data, 200

        except Exception as e:
            print(f"An error occurred: {e}")
            db.session.rollback()
            return {"message": "An error occurred while updating the stop"}, 400


@api.route("/operator-profiles/<int:stop_id>")
class OperatorDetail(Resource):
    """
    Endpoint to return a profile for each operator for a specific stop by its ID.
    """

    @api.response(404, "Not found")
    @api.response(400, "Validation Error")
    @api.response(503, "Unable to Respond")
    @api.response(200, "Successful")
    def get(self, stop_id):
        """
        GET method to return a profile for each operator who is operating a service departing from a desired stop within 90 minutes.

        Args:
            stop_id (int): The ID of the stop to retrieve details for.

        Returns:
            JSON: Details of the stop including the operator details.
        """
        try:
            # Retrieve the stop details from the database
            stop = Stops.query.filter_by(stop_id=stop_id).first()

            # Check if the stop exists
            if not stop:
                return {"message": "Stop not found"}, 404

            # Construct the URL to fetch departure data for the stop within 90 minutes
            url = (
                f"https://v6.db.transport.rest/stops/{stop_id}/departures?&duration=90"
            )
            response = requests.get(url, verify=False)

            # Check if the request was successful
            if response.ok:
                operator_data = response.json()
                unique_operator_names = set()  # Track unique operator names
                operator_names = []  # Initialise list for operator names

                # Extract unique operator names from the departure data
                for departure in operator_data["departures"]:
                    operator_name = departure["line"]["operator"]["name"]
                    if operator_name not in unique_operator_names:
                        unique_operator_names.add(operator_name)
                        operator_names.append(operator_name)
                        if len(operator_names) == 5:
                            break

                # If no operators found, return 404 with appropriate message
                if not operator_names:
                    return {"message": "No operators found"}, 404

                profiles = []

                # Generate a short summary for each operator using an external service
                for operator in operator_names:
                    question = f"provide a summary for the train operator: {operator}"
                    response = gemini.generate_content(question)
                    summary_text = response.text.replace("*", "").replace("\n", " ")
                    profiles.append(
                        {
                            "operator_name": operator,
                            "information": summary_text,
                        }
                    )

                # Return JSON response with stop ID and profiles
                return {
                    "stop_id": stop_id,
                    "profiles": profiles,
                }, 200
            else:
                # External service unavailable
                return {"message": "Unable to respond, Try again!"}, 503
        except Exception as e:
            # Handle unexpected errors gracefully
            print(f"An error occurred: {e}")
            return {"message": "Error, Try again!"}, 400


@api.route("/guide")
class TourismGuide(Resource):
    """
    Endpoint to create a tourism guide for exploring points of interest around a journey.
    """

    @api.response(400, "Validation Error")
    @api.response(404, "Journey Not Found")
    @api.response(503, "Unable to Respond")
    @api.response(200, "Successful")
    @api.doc(description="Generate a tourism guide in TXT file format.")
    def get(self):
        """
        GET method to generate a tourism guide in a TXT file format.

        Returns:
            TXT file: Tourism guide containing information about points of interest.
        """
        try:
            # Check if there are at least two stops in the database
            total_stops = Stops.query.count()
            if total_stops < 2:
                return {"message": "Not enough stops in the database"}, 400

            # Select two stops with a valid route between them
            source_stop = Stops.query.order_by(func.random()).first()
            destination_stop = (
                Stops.query.filter(Stops.stop_id != source_stop.stop_id)
                .order_by(func.random())
                .first()
            )

            # Check if there is a valid journey between source and destination stops
            journey_url = f"https://v6.db.transport.rest/journeys?from={source_stop.stop_id}&to={destination_stop.stop_id}"
            journey_response = requests.get(journey_url, verify=False)
            if not journey_response.ok:
                return {"message": "Journey not found"}, 404

            destination_names = []
            data = journey_response.json()

            for journey in data["journeys"]:
                for leg in journey["legs"]:
                    destination_name = leg["destination"]["name"]
                    destination_names.append(destination_name)

            # Select a destination name randomly from destination_names
            in_between_point_of_interest = random.choice(
                [
                    name
                    for name in destination_names
                    if name != source_stop.name and name != destination_stop.name
                ]
            )

            # Use Gemini API to generate content for source and destination
            source_content = gemini.generate_content(
                f"Tourist guide for {source_stop.name} as a minimum must include at least two points of interest (more if possible), tips for tourists, arrival and departure information, services and amenities available, souvenirs, popular food options, planning your visit, transportation options and any other relevant information (car rental, hotels and accommodation(list a few options), medical assistance(list a couple of options), sim-card and internet options, ATM and currency exchange and etc). provide some information about the weather, the best time to visit."
            )
            destination_content = gemini.generate_content(
                f"Tourist guide for {destination_stop.name} as a minimum must include at least two points of interest (more if possible), tips for tourists, arrival and departure information, services and amenities available, souvenirs, popular food options, planning your visit, transportation options and any other relevant information (car rental, hotels and accommodation(list a few options), medical assistance(list a couple of options), sim-card and internet options, ATM and currency exchange and etc). provide some information about the weather, the best time to visit."
            )

            # Generate content for in-between destination if available
            in_between_content = ""
            if in_between_point_of_interest:
                in_between_content = gemini.generate_content(
                    f"Tourist guide for {in_between_point_of_interest}"
                )

            # Create TXT file content with information about source, destination, and in-between destination
            txt_content = f"Tourism Guide\n\nSource: {source_stop.name}\n{source_content.text}\n\nDestination: {destination_stop.name}\n{destination_content.text}\n\nIn-between Point of Interest: {in_between_point_of_interest}\n{in_between_content.text}"

            # Save the TXT content to a temporary file
            file_name = f"{studentid}.txt"
            with open(file_name, "w") as file:
                file.write(txt_content)

            # Return the TXT file as an attachment
            return send_file(file_name, mimetype="text/plain", as_attachment=True)

        except Exception as e:
            print(f"An error occurred: {e}")
            return {"message": "Unable to generate tourism guide, Try again!"}, 503


class Stops(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    stop_type = db.Column(db.String, nullable=True)
    stop_id = db.Column(db.Integer, nullable=True)
    name = db.Column(db.String, nullable=True)
    latitude = db.Column(db.String, nullable=True)
    longitude = db.Column(db.String, nullable=True)
    products = db.Column(db.String, nullable=True)
    last_updated = db.Column(db.String, nullable=True)
    next_departure = db.Column(db.String, nullable=True)
    href = db.Column(db.String, nullable=True)


# NOTE REMOVE COMMENT ON THE FIRST RUN TO CREATE THE DB
# with app.app_context():
#     db.create_all()


if __name__ == "__main__":

    app.run(debug=True)
