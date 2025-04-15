from flask import Flask, request, render_template
import numpy as np
import pandas as pd

application=Flask(__name__)
app=application

def load_data():
    df=pd.read_csv