import streamlit as st
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

def scalers(df):
  df.dropna(inplace=True)
  scaler=st.selectbox("select scaler:",["StandardScaler", "MinMaxScaler","RobustScaler"])
  if scaler=="StandardScaler":
    df=pd.DataFrame(StandardScaler().fit_transform(df.select_dtypes(include="number")),columns=df.select_dtypes(include="number").columns).abs()
  elif scaler=="MinMaxScaler":
    df=pd.DataFrame(MinMaxScaler().fit_transform(df.select_dtypes(include="number")),columns=df.select_dtypes(include="number").columns).abs()
  elif scaler=="RobustScaler":
    df=pd.DataFrame(RobustScaler().fit_transform(df.select_dtypes(include="number")),columns=df.select_dtypes(include="number").columns).abs()
st.set_page_config(
    page_title="New app",
    page_icon="💻",
)
pages=["Home page", "EDA", "Modelling"]
st.sidebar.image("https://media-exp1.licdn.com/dms/image/C4D0BAQERT-S09jDMmg/company-logo_200_200/0/1636978765502?e=2147483647&v=beta&t=vQ2YUzkppflUhMHxqzLUinNnZhOXyFptaJodLYxnR04", width=420)
page=st.sidebar.selectbox("my app", pages)
st.title('my app')
upload_file = st.file_uploader('Upload Data file:')
try:
  df = pd.read_csv(upload_file)
  if page=="Home page":
    st.image("https://i0.wp.com/textilelearner.net/wp-content/uploads/2022/04/data-science.jpg?fit=604%2C359&ssl=1",width=800)
    st.write("Here's our dataframe:")
    st.write(df.head())
    st.header('Describe of Dataframe:')
    st.write(df.describe())
    st.header('Data frame columns:')
    st.write(df.columns)
    st.button("Re-run")
  elif page=="EDA":
    st.set_option('deprecation.showPyplotGlobalUse', False)
    column1,column2=st.columns(2)
    with column1:
      st.header('Data frame columns:')
      st.write(df.columns)
    with column2:
      st.header('Checking null value:')
      st.text(df.isnull().sum())
    st.header("Visualation data:")
    columns = df.columns.tolist()
    class_name = columns[-1]
    column_name = st.selectbox("",columns)
    plot_type = st.selectbox("", ["kde","box", "violin","swarm"])
    if st.button("Generate"):
      if plot_type == "kde":
          st.write(sns.FacetGrid(df, hue=class_name, palette="husl", height=6).map(sns.kdeplot, column_name).add_legend())
          st.pyplot()
      elif plot_type == "box":
          st.write(sns.boxplot(x=class_name, y=column_name, palette="husl", data=df))
          st.pyplot()
      elif plot_type == "violin":
          st.write(sns.violinplot(x=class_name, y=column_name, palette="husl", data=df))
          st.pyplot()
      elif plot_type == "swarm":
          st.write(sns.swarmplot(x=class_name, y=column_name, data=df,color="y", alpha=0.9))
          st.pyplot()
    if st.button("Outliers"):
      def detect_outliers_iqr(data):
          outliers = []
          data = sorted(data)
          q1 = np.percentile(data, 25)
          q3 = np.percentile(data, 75)
          IQR = q3-q1
          lwr_bound = q1-(1.5*IQR)
          upr_bound = q3+(1.5*IQR)
          for i in data:
              if (i < lwr_bound or i > upr_bound):
                  outliers.append(i)
          return outliers
      for i in df.select_dtypes(include="number").columns:
          sample_outliers = detect_outliers_iqr(df[i])
          st.text(f"Outliers of column {i} from IQR method: {len(sample_outliers)}")
    selected_column=st.selectbox("Inbalance checking", df.columns)
    st.write(df[selected_column].value_counts())
    x=st.selectbox("target for checking:", df.columns)
    y = st.selectbox("Select column",df.columns)
    plot_name = st.selectbox("", ["bar", "box"])
    if st.button("Inbalance graphic"):
      if plot_name == "box":
          st.write(df.plot(kind="box"))
          st.pyplot()
      elif plot_name == "bar":
          st.write(df.plot(kind="bar",x=x,y=y))
          st.pyplot()
    st.header('Imputation')  
    col1,col2,col3=st.columns(3)
    with col1:
      cat = st.radio(
        "Null Categircal value",
        ('Mode', 'Unknown', 'Drop'))
      cat_df=df.select_dtypes(exclude="number")
      if st.button("Run cat"):
        if cat == 'Mode':  
          for i in cat_df.columns:
            df[i].fillna(df[i].mode,inplace=True)
        elif cat=="Unknown":
          for i in cat_df.columns:
            df[i].fillna("Unknown",inplace=True)
        elif cat=="Drop":
          for i in cat_df.columns:
            df[i].dropna(inplace=True)
        else:
          st.write("You didn't select anyything fir categoric null value.")
    with col2:
      num = st.radio(
        "Null Numerical value",
        ('Mode', 'Median', 'Mean','Drop'))
      num_df=df.select_dtypes(include="number")
      if st.button("Run num"):
        if num == 'Mode':
          for i in num_df.columns:
            df[i].fillna(df[i].mode,inplace=True)
        elif num=="Median":
          for i in num_df.columns:
            df[i].fillna(df[i].median(),inplace=True)
        elif num=="Mean":
          for i in num_df.columns:
            df[i].fillna(df[i].mean(),inplace=True)
        elif num=="Drop":
          for i in num_df.columns:
            df[i].dropna(inplace=True)
        else:
          st.write("You didn't select anyything fir categoric null value.")
    with col3:
      st.text("Feature engineering")
      co = st.checkbox("Clean Outliers")
      if st.button("Run features"):
        if co:
          def replace_outlier(df_in, col_name):
            data=df_in[col_name]
            data = sorted(data)
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            IQR = q3-q1 
            lwr_bound  = q1-1.5*IQR
            upr_bound= q3+1.5*IQR
            df_out = np.clip(data, lwr_bound, upr_bound)
            return df_out
          for i in df.select_dtypes(include="number").columns:
              df[i]=replace_outlier(df, i)
  elif page=="Modelling":
    id=st.selectbox("Drop Id:", df.columns)
    encoder1,encoder2,encoder3=st.columns(3)
    with encoder1:
      if st.button("get dummies"):
        df.dropna(inplace=True)
        df=pd.get_dummies(df.drop(id,axis=1))
    with encoder2:
      if st.button("Label encoder"):
        df.dropna(inplace=True)
        df = LabelEncoder().fit_transform(df)
    with encoder3:
      if st.button("One encoder"):
        df.dropna(inplace=True)
        df = OneHotEncoder().fit_transform(df)
    st.write(df)
    scalers(df)
    st.write(df)
    i=st.selectbox("select target:",df.columns)
    X=pd.get_dummies(df.drop(i,axis=1))
    Y=df[i]
    X_train, X_test, y_train, y_test = train_test_split (X, Y, test_size = 0.2, random_state = 0)
    model=st.selectbox("Select model:",["LogisticRegression", "DecisionTreeClassifier", "GaussianNB"])
    if model=="LogisticRegression":
      lr=LogisticRegression()
      lr.fit(X_train,y_train)
      st.write(f"train score: {lr.score(X_train,y_train)}")
      st.write(f"train score: {lr.score(X_test,y_test)}")
      st.write(classification_report(y_test,lr.predict(X_test)))
    elif model=="DecisionTreeClassifier":
      tree=DecisionTreeClassifier()
      tree.fit(X_train,y_train)
      st.write(f"train score: {tree.score(X_train,y_train)}")
      st.write(f"train score: {tree.score(X_test,y_test)}")
      st.write(classification_report(y_test,tree.predict(X_test)))
    elif model=="GaussianNB":
      gb=GaussianNB()
      gb.fit(X_train,y_train)
      st.write(f"train score: {gb.score(X_train,y_train)}")
      st.write(f"test score: {gb.score(X_test,y_test)}")
      st.write(classification_report(y_test,gb.predict(X_test)))
    else:
      pass

except:
  if page=="Home page":
    st.write("Select dataframe")
  elif page=="EDA":
    st.write("Select dataframe")
  elif page=="Modelling":
    st.write("Select dataframe")