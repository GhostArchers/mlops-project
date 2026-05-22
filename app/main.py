from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import joblib
import numpy as np

app = FastAPI()

model    = joblib.load("models/model.pkl")
scaler   = joblib.load("models/scaler.pkl")
selector = joblib.load("models/selector.pkl")

@app.post("/predict")
def predict(data: dict):
    try:
        features = [
            data["Area"],
            data["Sensing Range"],
            data["Transmission Range"],
            data["Number of Sensor nodes"]
        ]
        X          = np.array(features).reshape(1, -1)
        X_scaled   = scaler.transform(X)
        X_selected = selector.transform(X_scaled)
        prediction = model.predict(X_selected)[0]
        return {"predicted_barriers": float(prediction)}
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
def health():
    return {"message": "Barrier Prediction API is running"}

@app.get("/", response_class=HTMLResponse)
def ui():
    return HTML

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Barrier Prediction</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: sans-serif;
    background: #f4f6f9;
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 100vh;
    padding: 2rem;
  }
  .card {
    background: white;
    border-radius: 12px;
    padding: 2rem;
    width: 100%;
    max-width: 460px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
  }
  h1 { font-size: 1.5rem; margin-bottom: 0.3rem; color: #1a1a2e; }
  p  { font-size: 0.85rem; color: #666; margin-bottom: 1.5rem; }
  label {
    display: block;
    font-size: 0.78rem;
    font-weight: 600;
    color: #444;
    margin-bottom: 0.3rem;
    margin-top: 0.9rem;
  }
  input {
    width: 100%;
    padding: 0.6rem 0.9rem;
    border: 1px solid #ddd;
    border-radius: 8px;
    font-size: 0.9rem;
    outline: none;
    transition: border-color 0.2s;
  }
  input:focus { border-color: #4f46e5; }
  button {
    width: 100%;
    margin-top: 1.4rem;
    padding: 0.75rem;
    background: #4f46e5;
    color: white;
    border: none;
    border-radius: 8px;
    font-size: 0.95rem;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.2s;
  }
  button:hover { background: #4338ca; }
  button:disabled { background: #a5b4fc; cursor: not-allowed; }
  #result {
    margin-top: 1.2rem;
    padding: 1rem;
    border-radius: 8px;
    display: none;
    text-align: center;
  }
  #result.ok  { display: block; background: #f0fdf4; border: 1px solid #86efac; }
  #result.err { display: block; background: #fef2f2; border: 1px solid #fca5a5; }
  .val  { font-size: 2rem; font-weight: 700; color: #16a34a; }
  .unit { font-size: 0.78rem; color: #666; margin-top: 0.2rem; }
  .errmsg { color: #dc2626; font-size: 0.85rem; }
</style>
</head>
<body>
<div class="card">
  <h1>Barrier Prediction</h1>
  <p>Enter sensor network parameters to predict coverage barriers.</p>

  <label>Area (m²)</label>
  <input type="number" id="area" placeholder="e.g. 100" step="any"/>

  <label>Sensing Range (m)</label>
  <input type="number" id="sensing" placeholder="e.g. 10" step="any"/>

  <label>Transmission Range (m)</label>
  <input type="number" id="transmission" placeholder="e.g. 20" step="any"/>

  <label>Number of Sensor Nodes</label>
  <input type="number" id="sensors" placeholder="e.g. 50" step="any"/>

  <button id="btn" onclick="go()">Predict</button>

  <div id="result">
    <div class="val"  id="rval">-</div>
    <div class="unit" id="runit"></div>
  </div>
</div>

<script>
async function go() {
  const area = parseFloat(document.getElementById('area').value);
  const sens = parseFloat(document.getElementById('sensing').value);
  const tran = parseFloat(document.getElementById('transmission').value);
  const sens2= parseFloat(document.getElementById('sensors').value);

  if ([area, sens, tran, sens2].some(isNaN)) {
    showErr("Please fill in all four fields.");
    return;
  }

  const btn = document.getElementById('btn');
  btn.disabled = true;
  btn.textContent = 'Predicting...';

  try {
    const res  = await fetch('/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        "Area": area,
        "Sensing Range": sens,
        "Transmission Range": tran,
        "Number of Sensor nodes": sens2
      })
    });
    const data = await res.json();
    if (data.predicted_barriers !== undefined) {
      const box = document.getElementById('result');
      box.className = 'ok';
      document.getElementById('rval').textContent  = data.predicted_barriers.toFixed(2);
      document.getElementById('runit').textContent = 'barriers predicted';
    } else {
      showErr(data.error || 'Prediction failed.');
    }
  } catch(e) {
    showErr('Could not reach the server.');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Predict';
  }
}

function showErr(msg) {
  const box = document.getElementById('result');
  box.className = 'err';
  document.getElementById('rval').innerHTML  = '<span class="errmsg">'+msg+'</span>';
  document.getElementById('runit').textContent = '';
}

document.addEventListener('keydown', e => { if (e.key === 'Enter') go(); });
</script>
</body>
</html>
"""