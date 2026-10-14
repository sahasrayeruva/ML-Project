
from flask import Flask, jsonify, request, send_from_directory
from pathlib import Path
import pandas as pd
import joblib

app = Flask(__name__)

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_FILE = BASE_DIR / "models" / "road_risk_model.pkl"
DATA_FILE = BASE_DIR / "data" / "hyderabad_ml_predictions.csv"
ACCIDENTS_FILE = BASE_DIR / "data" / "hyderabad_accidents_clean.csv"

# --------------------------------------------------
# LOAD ACCIDENT DATA
# --------------------------------------------------

accidents = pd.read_csv(ACCIDENTS_FILE)

# --------------------------------------------------
# LOAD MODEL AND ROAD DATA
# --------------------------------------------------

print("Loading LightGBM model...")
model = joblib.load(MODEL_FILE)
print("Model loaded successfully!")

print("Loading road data...")
roads = pd.read_csv(DATA_FILE)
print("Roads loaded:", len(roads))

# --------------------------------------------------
# HOME
# --------------------------------------------------

@app.route("/")
def home():
    return send_from_directory(
        BASE_DIR / "app",
        "index.html"
    )



# --------------------------------------------------
# DASHBOARD STATISTICS
# --------------------------------------------------

@app.route("/api/stats")
def stats():

    counts = roads["predicted_risk_level"].value_counts()

    return jsonify({
        "total_roads": int(len(roads)),
        "low": int(counts.get("Low", 0)),
        "medium": int(counts.get("Medium", 0)),
        "high": int(counts.get("High", 0)),
        "average_confidence": round(
            float(roads["prediction_confidence"].mean()),
            4
        )
    })


# --------------------------------------------------
# ROAD SEARCH
# --------------------------------------------------

@app.route("/api/search")
def search():

    query = request.args.get("q", "").strip()

    if not query:
        return jsonify([])

    # Find matching road segments
    matches = roads[
        roads["name"]
        .fillna("")
        .astype(str)
        .str.contains(query, case=False, regex=False)
    ].copy()

    if matches.empty:
        return jsonify([])

    # Risk priority
    risk_order = {
        "High": 3,
        "Medium": 2,
        "Low": 1
    }

    matches["risk_rank"] = (
        matches["predicted_risk_level"]
        .map(risk_order)
        .fillna(0)
    )

    # Group all segments having the same road name
    grouped_results = []

    for road_name, group in matches.groupby("name"):

        # Select the highest-risk segment
        highest_risk_row = group.sort_values(
            "risk_rank",
            ascending=False
        ).iloc[0]

        # Most common road type
        road_type = (
            group["highway"]
            .dropna()
            .astype(str)
            .mode()
        )

        road_type = (
            road_type.iloc[0]
            if not road_type.empty
            else "Unknown"
        )

        # Average speed
        speed = pd.to_numeric(
            group["maxspeed"],
            errors="coerce"
        ).mean()

        # Average lanes
        lanes = pd.to_numeric(
            group["lanes"],
            errors="coerce"
        ).mean()

        # Total road length
        length = pd.to_numeric(
            group["length"],
            errors="coerce"
        ).sum()

        # Total historical accidents
        accidents = pd.to_numeric(
            group["accident_count"],
            errors="coerce"
        ).fillna(0).sum()

        grouped_results.append({

            "name": str(road_name),

            "road_type": road_type,

            # Use the highest-risk segment
            "risk": str(
                highest_risk_row["predicted_risk_level"]
            ),

            "confidence": round(
                float(
                    highest_risk_row[
                        "prediction_confidence"
                    ]
                ),
                4
            ),

            "speed": (
                None
                if pd.isna(speed)
                else round(float(speed), 1)
            ),

            "lanes": (
                None
                if pd.isna(lanes)
                else round(float(lanes), 1)
            ),

            "length": round(
                float(length),
                2
            ),

            "accidents": int(
                round(accidents)
            )
        })

    # Sort High → Medium → Low
    grouped_results.sort(
        key=lambda x: risk_order.get(
            x["risk"],
            0
        ),
        reverse=True
    )

    # Return maximum 20 unique road names
    return jsonify(
        grouped_results[:20]
    )


# --------------------------------------------------
# HIGH / MEDIUM RISK ROADS
# --------------------------------------------------

@app.route("/api/roads")
def get_roads():

    important = roads[
        roads["predicted_risk_level"].isin(
            ["Medium", "High"]
        )
    ].copy()

    if important.empty:
        return jsonify([])

    # Risk priority
    risk_order = {
        "High": 3,
        "Medium": 2,
        "Low": 1
    }

    important["risk_rank"] = (
        important["predicted_risk_level"]
        .map(risk_order)
        .fillna(0)
    )

    results = []

    # Treat all segments with the same name as one road
    for road_name, group in important.groupby("name", dropna=False):

        # Ignore unnamed roads for the searchable road list
        if pd.isna(road_name):
            road_name = "Unnamed Road"

        # Select the highest-risk segment
        highest_risk_row = group.sort_values(
            "risk_rank",
            ascending=False
        ).iloc[0]

        # Most common road type
        road_type = (
            group["highway"]
            .dropna()
            .astype(str)
            .mode()
        )

        road_type = (
            road_type.iloc[0]
            if not road_type.empty
            else "Unknown"
        )

        # Average confidence
        confidence = pd.to_numeric(
            group["prediction_confidence"],
            errors="coerce"
        ).mean()

        # Average speed
        speed = pd.to_numeric(
            group["maxspeed"],
            errors="coerce"
        ).mean()

        # Average lanes
        lanes = pd.to_numeric(
            group["lanes"],
            errors="coerce"
        ).mean()

        # Total length of all segments
        length = pd.to_numeric(
            group["length"],
            errors="coerce"
        ).fillna(0).sum()

        # Average geometry features
        straightness = pd.to_numeric(
            group["straightness"],
            errors="coerce"
        ).mean()

        geometry_complexity = pd.to_numeric(
            group["geometry_complexity"],
            errors="coerce"
        ).mean()

        # IMPORTANT:
        # Sum accident history across ALL segments
        accidents = pd.to_numeric(
            group["accident_count"],
            errors="coerce"
        ).fillna(0).sum()

        results.append({

            "name": str(road_name),

            "road_type": road_type,

            # Highest risk among all segments
            "risk": str(
                highest_risk_row[
                    "predicted_risk_level"
                ]
            ),

            "confidence": round(
                float(confidence),
                4
            ) if not pd.isna(confidence) else 0,

            "speed": (
                None
                if pd.isna(speed)
                else round(float(speed), 1)
            ),

            "lanes": (
                None
                if pd.isna(lanes)
                else round(float(lanes), 1)
            ),

            "length": round(
                float(length),
                2
            ),

            "straightness": (
                None
                if pd.isna(straightness)
                else round(
                    float(straightness),
                    4
                )
            ),

            "geometry_complexity": (
                None
                if pd.isna(geometry_complexity)
                else round(
                    float(geometry_complexity),
                    4
                )
            ),

            "accidents": int(
                round(accidents)
            )
        })

    # High-risk roads first
    results.sort(
        key=lambda x: risk_order.get(
            x["risk"],
            0
        ),
        reverse=True
    )

    # Keep the response manageable
    return jsonify(results[:5000])


# --------------------------------------------------
# REAL LIGHTGBM PREDICTION API
# --------------------------------------------------

@app.route("/api/predict", methods=["POST"])
def predict():

    data = request.get_json()

    required_features = [
        "highway",
        "maxspeed",
        "oneway",
        "reversed",
        "length",
        "lanes",
        "geometry_points",
        "endpoint_distance",
        "geometry_complexity",
        "straightness"
    ]

    # Check input
    missing = [
        feature
        for feature in required_features
        if feature not in data
    ]

    if missing:
        return jsonify({
            "error": "Missing features",
            "missing": missing
        }), 400

    # Create one-row dataframe
    input_data = pd.DataFrame([data])

    # REAL MODEL PREDICTION
    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    risk_names = {
        0: "Low",
        1: "Medium",
        2: "High"
    }

    risk = risk_names[int(prediction)]

    confidence = float(
        max(probabilities)
    )

    return jsonify({

        "predicted_risk": risk,

        "confidence": round(
            confidence,
            4
        ),

        "probabilities": {
            "Low": round(
                float(probabilities[0]),
                4
            ),

            "Medium": round(
                float(probabilities[1]),
                4
            ),

            "High": round(
                float(probabilities[2]),
                4
            )
        }
    })


# --------------------------------------------------
# START SERVER
# --------------------------------------------------
@app.route("/api/accidents")
def get_accidents():
    data = accidents[
        [
            "accident_id",
            "latitude",
            "longitude",
            "date",
            "time",
            "weather",
            "cause",
            "accident_severity",
            "vehicles_involved",
            "casualties",
            "traffic_density"
        ]
    ].copy()

    data = data.dropna(subset=["latitude", "longitude"])

    return jsonify(data.to_dict(orient="records"))

if __name__ == "__main__":

    print("")
    print("==============================")
    print("HYDERABAD ROAD RISK AI")
    print("==============================")
    print("Model: LightGBM")
    print("Roads:", len(roads))
    print("")
    print("Server:")
    print("http://127.0.0.1:5000")
    print("")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )