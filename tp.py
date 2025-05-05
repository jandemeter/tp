import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from io import BytesIO
import math

# Načítanie FastAPI a ďalších knižníc
app = FastAPI()

# Načítanie modelu
model = joblib.load('multiclass_model_AD.pkl')
# label_encoder = joblib.load('label_encoder.pkl') 

def calculate_angle(diff_x, diff_y):
    angle_radians = math.atan2(diff_x, diff_y)
    angle_degrees = math.degrees(angle_radians)
    if angle_degrees < 0:
        angle_degrees += 360
    return angle_degrees

def calculate_tms(x1, y1, x2, y2, time1, time2):
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    time_diff = time2 - time1
    if time_diff > 0:
        return distance / time_diff
    else:
        return np.nan

def merge_files(touch_df, accelerometer_df, gyroscope_df):
    touch_df = touch_df.drop(columns=["event_type_detail", "pointer_id", "raw_x", "raw_y", "touch_major", "touch_minor"], errors='ignore')
    touch_df = touch_df.rename(columns={
        "event_type": "touch_event_type",
        "x": "touch_x",
        "y": "touch_y",
        "pressure": "touch_pressure",
        "size": "touch_size"
    })

    for col in ["userid", "vzor_id", "orientation"]:
        if col not in touch_df.columns:
            touch_df[col] = None 

    accelerometer_df = accelerometer_df.rename(columns={
        "x": "accelerometer_x", 
        "y": "accelerometer_y", 
        "z": "accelerometer_z"
    })

    gyroscope_df = gyroscope_df.rename(columns={
        "x": "gyroscope_x", 
        "y": "gyroscope_y", 
        "z": "gyroscope_z"
    })

    merged_df = touch_df.merge(
        accelerometer_df[["timestamp", "vzor_id", "accelerometer_x", "accelerometer_y", "accelerometer_z"]],
        on=["timestamp", "vzor_id"], how="left"
    )
    merged_df = merged_df.merge(
        gyroscope_df[["timestamp", "vzor_id", "gyroscope_x", "gyroscope_y", "gyroscope_z"]],
        on=["timestamp", "vzor_id"], how="left"
    )

    column_order = [
        "userid", "vzor_id", "timestamp", "orientation",
        "touch_event_type", "touch_x", "touch_y", "touch_pressure", "touch_size",
        "accelerometer_x", "accelerometer_y", "accelerometer_z",
        "gyroscope_x", "gyroscope_y", "gyroscope_z"
    ]

    for col in column_order:
        if col not in merged_df.columns:
            merged_df[col] = None

    merged_df = merged_df[column_order]

    return merged_df

processed_data = []
current_touch = None

def process_files(df, first_userid):
# priprava na predspracovanie (vypocet angle, direction, TMS)
    for _, row in df.iterrows():
        if row['touch_event_type'] == 'down':
            current_touch = {
                "userid": row.get("userid", np.nan),
                "timestamp": row["timestamp"],
                "touch_event_type": row["touch_event_type"],
                "touch_x": row["touch_x"],
                "touch_y": row["touch_y"],
                "direction": np.nan,
                "angle": np.nan,
                "touch_pressure": row["touch_pressure"],
                "touch_size": row["touch_size"]
            }
        elif row['touch_event_type'] in ['move', 'up'] and current_touch:
            # Zistíme, či sa pozícia zmenila
            diff_x = row["touch_x"] - current_touch["touch_x"]
            diff_y = row["touch_y"] - current_touch["touch_y"]

            if diff_x == 0 and diff_y == 0 and row['touch_event_type'] == 'move':
                continue
            elif diff_x == 0 and diff_y == 0:
                direction = np.nan
                angle = np.nan
                TMS = np.nan
            else:
                angle = calculate_angle(diff_x, diff_y)
                TMS = calculate_tms(current_touch["touch_x"], current_touch["touch_y"], row["touch_x"], row["touch_y"],
                                    current_touch["timestamp"], row["timestamp"])

                match angle:
                    case angle if 0 <= angle < 45:
                        direction = 1
                    case angle if 45 <= angle < 90:
                        direction = 2
                    case angle if 90 <= angle < 135:
                        direction = 3
                    case angle if 135 <= angle < 180:
                        direction = 4
                    case angle if 180 <= angle < 225:
                        direction = 5
                    case angle if 225 <= angle < 270:
                        direction = 6
                    case angle if 270 <= angle < 315:
                        direction = 7
                    case _:
                        direction = 8

            current_touch = {
            "userid": row.get("userid", np.nan),
            "timestamp": row["timestamp"],
            "touch_event_type": row["touch_event_type"],
            "touch_x": row["touch_x"],
            "touch_y": row["touch_y"],
            "touch_pressure": row["touch_pressure"],
            "touch_size": row["touch_size"],
            "accelerometer_x": row["accelerometer_x"],
            "accelerometer_y": row["accelerometer_y"],
            "accelerometer_z": row["accelerometer_z"],
            "gyroscope_x": row["gyroscope_x"],
            "gyroscope_y": row["gyroscope_y"],
            "gyroscope_z": row["gyroscope_z"],
            "direction": direction,
            "angle": angle,
            "TMS": TMS,
        }

        processed_data.append(current_touch)


    processed_df = pd.DataFrame(processed_data)


    
    if not processed_df.empty:
        processed_df['userid'] = processed_df['userid'].fillna(first_userid)

    return processed_df

def create_features(df):
    data = []

    # inicializacia premennych
    for _, user_data in df.groupby('userid'):
        movement_data = None
        direction_data = {i: [] for i in range(1, 9)}
        length_data = {i: 0 for i in range(1, 9)}
        acceleration_x = {i: [] for i in range(1, 9)}
        acceleration_y = {i: [] for i in range(1, 9)}
        acceleration_z = {i: [] for i in range(1, 9)}
        total_acceleration = {i: [] for i in range(1, 9)}
        gyro_x = {i: [] for i in range(1, 9)}
        gyro_y = {i: [] for i in range(1, 9)}
        gyro_z = {i: [] for i in range(1, 9)}
        total_gyro = {i: [] for i in range(1, 9)}

        prev_x, prev_y = None, None

        # vytvaranie features
        for _, row in user_data.iterrows():
            # zaciatok pohybu
            if row["touch_event_type"] == "down":
                movement_data = {"userid": row["userid"]}
                direction_data = {i: [] for i in range(1, 9)}
                length_data = {i: 0 for i in range(1, 9)}
                acceleration_x = {i: [] for i in range(1, 9)}
                acceleration_y = {i: [] for i in range(1, 9)}
                acceleration_z = {i: [] for i in range(1, 9)}
                total_acceleration = {i: [] for i in range(1, 9)}
                gyro_x = {i: [] for i in range(1, 9)}
                gyro_y = {i: [] for i in range(1, 9)}
                gyro_z = {i: [] for i in range(1, 9)}
                total_gyro = {i: [] for i in range(1, 9)}
                prev_x, prev_y = row["touch_x"], row["touch_y"]

            # priebeh pohybu
            elif row["touch_event_type"] == "move" and movement_data:
                direction = row["direction"]
                if direction in range(1, 9):
                    direction_data[direction].append(row["TMS"])

                    if prev_x is not None and prev_y is not None:
                        length_data[direction] += np.sqrt(
                            (row["touch_x"] - prev_x) ** 2 + (row["touch_y"] - prev_y) ** 2)

                    acceleration_x[direction].append(row["accelerometer_x"])
                    acceleration_y[direction].append(row["accelerometer_y"])
                    acceleration_z[direction].append(row["accelerometer_z"])
                    total_acceleration[direction].append(np.sqrt(row["accelerometer_x"] ** 2 + row["accelerometer_y"] ** 2 + row["accelerometer_z"] ** 2))

                    gyro_x[direction].append(row["gyroscope_x"])
                    gyro_y[direction].append(row["gyroscope_y"])
                    gyro_z[direction].append(row["gyroscope_z"])
                    total_gyro[direction].append(np.sqrt(row["gyroscope_x"] ** 2 + row["gyroscope_y"] ** 2 + row["gyroscope_z"] ** 2))

                    prev_x, prev_y = row["touch_x"], row["touch_y"]

            # koniec pohybu
            elif row["touch_event_type"] == "up" and movement_data:
                for direction in range(1, 9):
                    movement_data[f"ATMS_{direction}"] = round(np.mean(direction_data[direction]), 20) if direction_data[direction] else np.nan
                    movement_data[f"max_TMS_{direction}"] = round(np.max(direction_data[direction]), 20) if direction_data[direction] else np.nan
                    movement_data[f"min_TMS_{direction}"] = round(np.min(direction_data[direction]), 20) if direction_data[direction] else np.nan
                    movement_data[f"length_{direction}"] = round(length_data[direction], 6) if length_data[direction] > 0 else np.nan
                    movement_data[f"accel_x_{direction}"] = round(np.mean(acceleration_x[direction]), 6) if acceleration_x[direction] else np.nan
                    movement_data[f"accel_y_{direction}"] = round(np.mean(acceleration_y[direction]), 6) if acceleration_y[direction] else np.nan
                    movement_data[f"accel_z_{direction}"] = round(np.mean(acceleration_z[direction]), 6) if acceleration_z[direction] else np.nan
                    movement_data[f"max_accel_x_{direction}"] = round(np.max(acceleration_x[direction]), 6) if acceleration_x[direction] else np.nan
                    movement_data[f"min_accel_x_{direction}"] = round(np.min(acceleration_x[direction]), 6) if acceleration_x[direction] else np.nan
                    movement_data[f"max_accel_y_{direction}"] = round(np.max(acceleration_y[direction]), 6) if acceleration_y[direction] else np.nan
                    movement_data[f"min_accel_y_{direction}"] = round(np.min(acceleration_y[direction]), 6) if acceleration_y[direction] else np.nan
                    movement_data[f"max_accel_z_{direction}"] = round(np.max(acceleration_z[direction]), 6) if acceleration_z[direction] else np.nan
                    movement_data[f"min_accel_z_{direction}"] = round(np.min(acceleration_z[direction]), 6) if acceleration_z[direction] else np.nan
                    movement_data[f"total_accel_{direction}"] = round(np.mean(total_acceleration[direction]), 6) if total_acceleration[direction] else np.nan
                    movement_data[f"gyro_x_{direction}"] = round(np.mean(gyro_x[direction]), 6) if gyro_x[direction] else np.nan
                    movement_data[f"gyro_y_{direction}"] = round(np.mean(gyro_y[direction]), 6) if gyro_y[direction] else np.nan
                    movement_data[f"gyro_z_{direction}"] = round(np.mean(gyro_z[direction]), 6) if gyro_z[direction] else np.nan
                    movement_data[f"max_gyro_x_{direction}"] = round(np.max(gyro_x[direction]), 6) if gyro_x[direction] else np.nan
                    movement_data[f"min_gyro_x_{direction}"] = round(np.min(gyro_x[direction]), 6) if gyro_x[direction] else np.nan
                    movement_data[f"max_gyro_y_{direction}"] = round(np.max(gyro_y[direction]), 6) if gyro_y[direction] else np.nan
                    movement_data[f"min_gyro_y_{direction}"] = round(np.min(gyro_y[direction]), 6) if gyro_y[direction] else np.nan
                    movement_data[f"max_gyro_z_{direction}"] = round(np.max(gyro_z[direction]), 6) if gyro_z[direction] else np.nan
                    movement_data[f"min_gyro_z_{direction}"] = round(np.min(gyro_z[direction]), 6) if gyro_z[direction] else np.nan
                    movement_data[f"total_gyro_{direction}"] = round(np.mean(total_gyro[direction]), 6) if total_gyro[direction] else np.nan

                data.append(movement_data)
                movement_data = None

    df_out = pd.DataFrame(data)
    columns_order = ["userid"] + \
                    [f"ATMS_{i}" for i in range(1, 9)] + \
                    [f"max_TMS_{i}" for i in range(1, 9)] + \
                    [f"min_TMS_{i}" for i in range(1, 9)] + \
                    [f"length_{i}" for i in range(1, 9)] + \
                    [f"accel_x_{i}" for i in range(1, 9)] + \
                    [f"accel_y_{i}" for i in range(1, 9)] + \
                    [f"accel_z_{i}" for i in range(1, 9)] + \
                    [f"max_accel_x_{i}" for i in range(1, 9)] + \
                    [f"min_accel_x_{i}" for i in range(1, 9)] + \
                    [f"max_accel_y_{i}" for i in range(1, 9)] + \
                    [f"min_accel_y_{i}" for i in range(1, 9)] + \
                    [f"max_accel_z_{i}" for i in range(1, 9)] + \
                    [f"min_accel_z_{i}" for i in range(1, 9)] + \
                    [f"total_accel_{i}" for i in range(1, 9)] + \
                    [f"gyro_x_{i}" for i in range(1, 9)] + \
                    [f"gyro_y_{i}" for i in range(1, 9)] + \
                    [f"gyro_z_{i}" for i in range(1, 9)] + \
                    [f"max_gyro_x_{i}" for i in range(1, 9)] + \
                    [f"min_gyro_x_{i}" for i in range(1, 9)] + \
                    [f"max_gyro_y_{i}" for i in range(1, 9)] + \
                    [f"min_gyro_y_{i}" for i in range(1, 9)] + \
                    [f"max_gyro_z_{i}" for i in range(1, 9)] + \
                    [f"min_gyro_z_{i}" for i in range(1, 9)] + \
                    [f"total_gyro_{i}" for i in range(1, 9)]

    return df_out[columns_order]

@app.post("/process/")
async def process_data(
    touch: UploadFile = File(...),
    accelerometer: UploadFile = File(...),
    gyroscope: UploadFile = File(...),
):
    try:
        touch_df = pd.read_csv(BytesIO(await touch.read()))
        accelerometer_df = pd.read_csv(BytesIO(await accelerometer.read()))
        gyroscope_df = pd.read_csv(BytesIO(await gyroscope.read()))

        if 'userid' in touch_df.columns and not touch_df['userid'].isnull().all():
            first_userid = touch_df['userid'].dropna().iloc[0]
        else:
            first_userid = None

        merged_df = merge_files(touch_df, accelerometer_df, gyroscope_df)
        processed_df = process_files(merged_df, first_userid)
        final_features_df = create_features(processed_df)

        if final_features_df.empty:
            raise ValueError("The features DataFrame is empty!")

        features = final_features_df.drop(columns=['userid', 'timestamp'], errors='ignore')

        if features.empty:
            raise ValueError("The features DataFrame is empty after dropping 'userid' and 'timestamp'!")

        predictions = model.predict(features)

        final_features_df['prediction'] = predictions
        final_features_df = final_features_df.replace([np.inf, -np.inf], np.nan)
        final_features_df = final_features_df.fillna(0)

        match_result = False

        result = {
            "match": match_result,
            "predictions": int(predictions[-1])
        }

        # VÝPIS PREDIKCIE DO LOGU
        print("Prediction result:", result)

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
