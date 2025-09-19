from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from backend_step_by_step import run_simulation
from models import EconomicParameters, FinancialParameters, Results, TechnicalParameters

app = FastAPI()
# Servir les fichiers statiques React
app.mount("/static", StaticFiles(directory="static/build/static"), name="static")

# Route catch-all pour React Router
@app.get("/{full_path:path}")
async def serve_react_app(full_path: str):
    index_path = os.path.join("static", "build", "index.html")
    return FileResponse(index_path)

@app.get("/tests")
def hello_world():
    return {"message : Hello World"}

@app.post("/calculate")
def calculate(technical_parameters : TechnicalParameters, 
              economic_parameters : EconomicParameters,
              financial_parameters : FinancialParameters ) -> Results:
    
    
    simulation_values = run_simulation(source = "espinoza", d = economic_parameters.d/100,
                            rom=economic_parameters.rOM/100,
                            rps=economic_parameters.rps/100,
                            rd=technical_parameters.rd/100, 
                            rpg=economic_parameters.rpg/100, 
                            P=technical_parameters.P,
                            Cu=economic_parameters.Cu,
                            Hopt=technical_parameters.Hopt, 
                            PR=technical_parameters.PR/100,
                            SCI=technical_parameters.SCI/100,
                            COM=economic_parameters.COM/100,
                            N=technical_parameters.N,
                            Xl=financial_parameters.Xl/100,
                            Xec=financial_parameters.Xec/100,
                            il=financial_parameters.il/100,
                            dec=financial_parameters.il/100,
                            T=economic_parameters.T/100,
                            Xd=economic_parameters.Xd/100,
                            Nd=economic_parameters.Nd,
                            Xis=financial_parameters.Xis/100,
                            Nis=financial_parameters.Nis,
                            Nl=financial_parameters.Nl,
                            ps=economic_parameters.ps,
                            pg=economic_parameters.pg)
    
    msg = f"Your data has been processed"
    result = Results(Result=simulation_values, msg=msg)
    return result

app.add_middleware(CORSMiddleware, 
    allow_origins = ["*"], allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],)