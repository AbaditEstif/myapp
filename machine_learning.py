import streamlit as st 

import pandas as pd 
import numpy as np 
import plotly.express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt 
import seaborn as sns
from io import BytesIO
import io
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
from sklearn.preprocessing  import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay


st.title( " predicting customer behavior using machine learning algorithm")

st.info('**predicting customer behavior using machine learning algorithm**')


with st.expander('**Data**'):
	st.write('**Raw data**')
	train= pd.read_csv( "data\Train.csv")
	test= pd.read_csv("data\Test.csv")
	data= pd.concat([train,test],ignore_index=True)
	st.dataframe(data.head())
	st.write('**Data Understanding**')

	data_shape= st.checkbox('show data sahpe')
	data_info=st.checkbox('show data info')
	if data_shape:
	 	st.write('Data shape:', data.shape)

	if data_info:
	 	st.write('Data info:')
	 	buffer= st.empty()
	 	data_info= data.info(buf=buffer)


	st.write('**You can dowload the dataset by clicking the button**')
	df= data
	csv_data = df.to_csv(index=False)
	st.download_button(label="Download CSV", 
	 	data=csv_data, 
	 	file_name='my_data.csv', mime='text/csv')


	st.write('**Clean data**')

	data['Family_Size'].fillna(data['Family_Size'].median())
	data['Work_Experience'].fillna(data['Work_Experience'].median())
	data['Graduated'].fillna(data['Graduated'].mode()[0])
	data['Ever_Married'].fillna(data['Ever_Married'].mode()[0])
	data['Profession'].fillna(data['Profession'].mode()[0])
	data['Var_1'].fillna(data['Var_1'].mode()[0])
	data.drop(["ID"],axis=1)
	st.write("DataFrame after filling missing values", data)

	
	
	# Remove outliers

	Q3= np.percentile(data['Work_Experience'], 75, method ='midpoint')
	Q1 = np.percentile(data['Work_Experience'], 25 , method ='midpoint')
	IQR= Q3-Q1
	upper =Q3+1.5*IQR
	upper_array =np.array(data['Work_Experience']>= upper)
	lower= Q1-1.5*IQR
	lower_array = np.array(data['Work_Experience']<= lower)
	data['Work_Experience']= data['Work_Experience'].apply(lambda x: lower if x<lower else(upper if x>upper else x))
	

	st.dataframe(data.head())

	st.write('**x input features**')
	x_raw=data.drop('Segmentation',axis=1)
	st.write(x_raw.head())

	st.write('**y target variable**')
	y_raw=data.Segmentation
	st.write(y_raw.head())




with st.expander("**Data Visualization**"):
	st.write('**Data Summary**')
	st.table(data.describe())
	st.write('**Univariate Analysis**')
	numerical_columns = data.select_dtypes(include=['number']).columns.tolist()
	if numerical_columns:
		column = st.selectbox("Select a numerical column for histogram:", numerical_columns)
		if column:
			st.write(f"#### Histogram of {column}")
			fig, ax = plt.subplots()
			sns.histplot(data[column], kde=True, ax=ax)
			st.pyplot(fig)
	st.write("### Value Counts")
	st.write("### Value Counts")
	categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()

	if categorical_columns:
		categorical_column = st.selectbox("Select a categorical column for value counts:", categorical_columns)
		if categorical_column:
			value_counts = data[categorical_column].value_counts()
			st.write(f"Value counts for column '{categorical_column}':")
			st.write(value_counts)
	fig, ax = plt.subplots()
	value_counts.plot(kind='bar', ax=ax)
	st.pyplot(fig)

	st.write('**Bivariate Analysis**')
	st.title('Segmentation Column vs. Other Columns')

	# Multiselect to select columns to plot against segmentation
	options=['Gender','Ever_Married','Age','Graduated','Work_Experience','Spending_Score','Family_Size','Var_1','Profession']
	selected_column = st.selectbox('Select column to plot against segment', options)
	fig = px.bar(data, x='Segmentation',  color=selected_column, barmode='group', 
             title=f'Segment vs. {selected_column} colored by {selected_column}')
	st.plotly_chart(fig)



	# Input features
	with st.sidebar:
	
    
		st.header('Input features')
		Gender=st.selectbox('Gender',('Male','Female'))
		Ever_Married=st.selectbox('Ever_Married',('Yes','No'))
		Age= st.slider('How old are you?',18,89,1)
		Graduated=st.selectbox('Graduated',('Yes','No'))

		Profession=st.selectbox('Profession',('Healthcare ','Entertainment','Engineer','Doctor',
			'Lawyer','Executive','Marketing','Marketing'))
		Work_Experience= st.slider('Work_Experience', 0.0,7.5,0.5)
		Spending_Score= st.selectbox('Spending_Score',('Low','Average','High'))
		Family_Size= st.slider('Family_Size',1,9,1)
		Var_1 = st.selectbox('Var_1',('Cat_1','Cat_2','Cat_3','Cat_4','Cat_5','Cat_6','Cat_7'))

		# create a Dataframe for the input features
		df={'Gender':Gender, 'Ever_Married':Ever_Married,'Age':Age,'Graduated':Graduated,
		      'Profession':Profession,'Work_Experience':Work_Experience,'Spending_Score':Spending_Score,
		        'Family_Size':Family_Size,'Var_1':Var_1}
		input_df = pd.DataFrame(df,index=[0])
		input_data= pd.concat([input_df,x_raw],axis=0)

		# Data Preparation    	    	       	   
	  
		# Feature Engineering
		
		df_customer =input_data[['Profession','Var_1','Gender','Graduated','Ever_Married']].apply(LabelEncoder().fit_transform)

		encoder =OrdinalEncoder(categories=[['Low', 'Average','High']])

		df_customer['Spending_Score']= encoder.fit_transform(input_data[['Spending_Score']])
		x= df_customer[1:]
		input_raw=df_customer[:1]

	

	# encode the target variable 

	target_mapper= {'A':0,'B':1,'C':2,'D':3}
	def target_encode(val):
		return target_mapper[val]
	y= y_raw.apply(target_encode)
							
with st.expander('**Input features**'):
	st.write('**Input features**')
	input_df
	st.write('**Combined input features**')
	input_data

with st.expander('**Data Preparation**'):
	st.write('** Encoded input features**')
	input_raw
	st.write('**Encoded target variable**')
	y

	# Model training

	clf= RandomForestClassifier()
	clf.fit(x,y)


	# Apply model to make prediction

	prediction= clf.predict(input_raw)
	pred_proba= clf.predict_proba(input_raw)
	

	df_pred_proba= pd.DataFrame(pred_proba)
	df_pred_proba.columns= ['A','B','C','D']
	df_pred_proba.rename(columns={0:'A',1:'B',2:'C',3:'D'})
	

	# Display predicted Segmentation
with st.expander('**Model Prediction**'):

	st.subheader('Predicted Segments')

	st.dataframe(df_pred_proba,column_config={
		'A':st.column_config.ProgressColumn(

			'A',format='%.2f',
			width= 'medium',
			min_value=0,
			max_value=1
			),
		'B':st.column_config.ProgressColumn(

			'B',format='%.2f',
			width= 'medium',
			min_value=0,
			max_value=1),
		'C':st.column_config.ProgressColumn(

			'C',format='%.2f',
			width= 'medium',
			min_value=0,
			max_value=1),
		'D':st.column_config.ProgressColumn(

			'D',format='%.2f',
			width= 'medium',
			min_value=0,
			max_value=1),

		} , hide_index= True)

	

	customer_segments= np.array(['A','B','C','D'])
	st.success(str(customer_segments[prediction][0]))


with st.expander('**Model Evaluation**'):
	st.write('Evaluate Model performance')

	y_pred= clf.predict(x)

	# calcualte accuracy
	accuracy = accuracy_score(y, y_pred)

	# Generate confusion matrix

	conf_matrix = confusion_matrix(y, y_pred)

	# Create a selectbox for metrics

	option = st.selectbox(

			"**Select metric to display**:",

			("Accuracy", "Confusion Matrix")

			)

	# Display the selcted metric
	if option == "Accuracy":
		st.write(f"Accuracy: {accuracy:.2f}")
	elif option == "Confusion Matrix":
		st.write("Confusion Matrix:")
		fig, ax = plt.subplots()
		sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
		st.pyplot(fig)


with st.expander('**Generate reports**'):
	st.write('**here are some insights**')
	def save_plot_to_bytes(fig):
		buf = io.BytesIO()
		fig.savefig(buf, format='png')
		buf.seek(0)
		return buf
	selected_column = st.selectbox(
		'Select the column to plot against Segmentation:',
		['Age', 'Family_Size', 'Work_Experience']
		)
	# Generate plots based on selected column
	fig, ax = plt.subplots()
	sns.violinplot(x='Segmentation', y=selected_column, data=data, ax=ax)
	ax.set_title(f'{selected_column} vs Segmentation')


	# save the plots to a BytesIo object
	buf = save_plot_to_bytes(fig)
	plt.close(fig)

	# Display the plot
	st.pyplot(fig)

	# add additional report
	report=f"""

	### report for {selected_column} vs Segmentation
	The violin plot above illustrates the distribution of `{selected_column}` across different segments.
	- **Segment D**:   for this segment the violin is wide and symmetric arround the median indicates, large concentration of age values near the medain.Segment D has younger customers with median age 20-40
	- **segments A&B**:  the violin plt for A &B show bimodal distribution, this indicates these two segments iclude both younger and older customers. The customer  median age ranges betwen 30-50 
	- **segment C**: segmet C also shows bimodal distribution  with median age closer to 50
	- **work experience**: the population has a work experience of less the 2 years across all segments 
	- **work experience**: the median is identical over all segments which indicates work experience is the same over all segments
	- **Family_Size**:The family size is less 5 with a spike at 2 over all segments. segment D follow uniform distribution till 4 with median closer to 3.Segment A&B has median closer to 2. segment A has the lowest rage. segment C has a spike at 2 with median closer to 2


	"""
	st.write(report)




	st.download_button(
		label="Download Plot",
		data=buf,
		file_name=f"{selected_column}_vs_segmentation.png",
		 mime="image/png"
		 )










		




	

     







































	

	



	

	
	

	 
	


	









		






    




	
	
	







			
			
	
		 
		        
		        






		
		
		


        








		                          








	




	

			

			
	    	






    

      
	    
	    	
	    	






	     

		    
		    

		    




			






	
	
		

		
	


    

 	


	
    




    
    
