import json
import os
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)

BASE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE, "modelos", "config.json"), encoding="utf-8") as f:
    CONFIG = json.load(f)

# Nota: los modelos entrenados se guardaron como .pkl (modelos/modelo_*.pkl).
# Una regresión lineal es y = b0 + b1*x1 + ..., así que la web usa el intercepto y
# los coeficientes exportados en config.json. Evita instalar scikit-learn (muy pesado
# para Vercel) y da exactamente el mismo resultado que model.predict().

HTML = """
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Predicciones con Regresión Lineal</title>
<style>
  *{box-sizing:border-box}
  body{font-family:system-ui,-apple-system,sans-serif;background:#eef2f7;margin:0;padding:24px;color:#1f2937}
  .card{max-width:500px;margin:auto;background:#fff;padding:28px;border-radius:14px;box-shadow:0 4px 16px #0002}
  h1{font-size:22px;margin:0 0 4px}
  .sub{color:#6b7280;margin:0 0 18px;font-size:14px}
  label{display:block;margin-top:14px;font-weight:600;font-size:14px}
  input,select{width:100%;padding:11px;margin-top:5px;border:1px solid #cbd5e1;border-radius:8px;font-size:16px}
  small{color:#6b7280;font-size:12px}
  button{margin-top:22px;width:100%;padding:13px;background:#2563eb;color:#fff;border:0;border-radius:8px;font-size:16px;cursor:pointer}
  button:hover{background:#1d4ed8}
  #res{margin-top:22px;padding:16px;border-radius:10px;background:#ecfdf5;color:#065f46;text-align:center;display:none}
  #res .v{font-size:28px;font-weight:700}
  #res.err{background:#fef2f2;color:#991b1b}
  .aviso{margin-top:8px;font-size:13px;color:#92400e}
  .ecu{margin-top:18px;font-size:12px;color:#6b7280;word-break:break-word}
</style>
</head>
<body>
<div class="card">
  <h1>Predicción con Regresión Lineal</h1>
  <p class="sub">Laboratorio de Minería de Datos</p>

  <label for="ej">Selecciona el ejercicio</label>
  <select id="ej" onchange="dibujar()"></select>

  <div id="campos"></div>
  <button onclick="predecir()">Predecir</button>
  <div id="res"></div>
  <div class="ecu" id="ecu"></div>
</div>

<script>
const CFG = {{ cfg|tojson }};
const sel = document.getElementById("ej");
for (const k in CFG) sel.innerHTML += `<option value="${k}">${CFG[k].titulo}</option>`;

function dibujar(){
  const m = CFG[sel.value];
  let h = "";
  m.features.forEach(f => {
    const [mn, mx] = m.rango[f];
    h += `<label for="in_${f}">${f}</label>
          <input type="number" step="any" id="in_${f}" placeholder="Ej: ${((mn+mx)/2).toFixed(2)}">
          <small>Rango en los datos de entrenamiento: ${mn.toFixed(2)} a ${mx.toFixed(2)}</small>`;
  });
  document.getElementById("campos").innerHTML = h;
  document.getElementById("res").style.display = "none";
  let e = `${m.target} = ${m.intercept.toFixed(4)}`;
  m.features.forEach((f,i) => e += ` ${m.coef[i]>=0?"+":"-"} ${Math.abs(m.coef[i]).toFixed(4)}·${f}`);
  document.getElementById("ecu").innerText = "Modelo: " + e + `  |  R² = ${m.r2.toFixed(3)}, RMSE = ${m.rmse.toFixed(2)}`;
}

async function predecir(){
  const m = CFG[sel.value];
  const valores = {};
  for (const f of m.features) valores[f] = document.getElementById("in_"+f).value;
  const r = await fetch("/predecir", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ejercicio: sel.value, valores})
  });
  const d = await r.json();
  const box = document.getElementById("res");
  box.style.display = "block";
  if (d.error){
    box.className = "err"; box.innerHTML = "⚠ " + d.error;
  } else {
    box.className = "";
    box.innerHTML = `<div>${d.target}</div><div class="v">${d.prediccion} ${d.unidad}</div>` +
      (d.avisos.length ? `<div class="aviso">⚠ ${d.avisos.join("<br>⚠ ")}</div>` : "");
  }
}
dibujar();
</script>
</body>
</html>
"""


@app.get("/")
def index():
    return render_template_string(HTML, cfg=CONFIG)


@app.post("/predecir")
def predecir():
    try:
        d = request.get_json()
        m = CONFIG[d["ejercicio"]]
        x = [float(d["valores"][f]) for f in m["features"]]
    except (KeyError, ValueError, TypeError):
        return jsonify(error="Completa todos los campos con números válidos."), 400

    y = m["intercept"] + sum(c * v for c, v in zip(m["coef"], x))
    avisos = []
    for f, v in zip(m["features"], x):
        mn, mx = m["rango"][f]
        if v < mn or v > mx:
            avisos.append(f"{f} está fuera del rango de entrenamiento ({mn:.2f} a {mx:.2f}); la predicción puede ser poco fiable.")
    return jsonify(prediccion=round(y, 2), unidad=m["unidad"], target=m["target"], avisos=avisos)


if __name__ == "__main__":
    app.run(debug=True)
