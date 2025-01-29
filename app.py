import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objs as go

# Teams and cities data
teams = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore', 'Kolkata Knight Riders', 
         'Kings XI Punjab', 'Chennai Super Kings', 'Rajasthan Royals', 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi', 'Chandigarh', 'Jaipur', 
          'Chennai', 'Cape Town', 'Port Elizabeth', 'Durban', 'Centurion', 'East London', 
          'Johannesburg', 'Kimberley', 'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 
          'Dharamsala', 'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi', 'Sharjah', 
          'Mohali', 'Bengaluru']

# Load the machine learning model
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Streamlit app title
st.title('IPL Win Predictor')

# Create columns using the updated method
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

selected_city = st.selectbox('Select host city', sorted(cities))

target = st.number_input('Target')

col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Score')
with col4:
    overs = st.number_input('Overs completed')
with col5:
    wickets = st.number_input('Wickets Out')

if st.button('Predict Probability'):
    runs_left = target - score
    balls_left =  120 - (overs * 6)
    wickets = 10 - wickets
    crr = score / overs
    rrr = (runs_left * 6) / balls_left

    # Create input DataFrame inside the button click block
    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [wickets],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    st.table(input_df)
    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]
    
    st.text(batting_team + "-" + str(round(win * 100, 2)) + "%")
    st.text(bowling_team + "-" + str(round(loss * 100, 2)) + "%")

    # Create pie chart using Plotly
    labels = [batting_team, bowling_team]
    sizes = [win * 100, loss * 100]

    fig = go.Figure(data=[go.Pie(labels=labels, values=sizes)])
    fig.update_traces(hoverinfo='label+percent', textinfo='value+percent', textfont_size=20,
                      marker=dict(colors=['#ff9999', '#66b3ff'], line=dict(color='#000000', width=2)))
    fig.update_layout(title_text="Win Probability", title_font_size=30, title_font_color="white", 
                      paper_bgcolor='black', font=dict(color="white"))
    
    # Format percentage to display only two decimal places
    fig.update_traces(texttemplate='%{value:.2f}%')

    st.plotly_chart(fig)
