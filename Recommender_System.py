import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from surprise import Reader, Dataset
from surprise.model_selection import train_test_split
from surprise import SVD

def get_recommendations(df, hotel_id, cosine_sim, nums=5):
    # Get the index of the hotel that matches the hotel_id
    matching_indices = df.index[df['Hotel_ID'] == hotel_id].tolist()
    if not matching_indices:
        print(f"No hotel found with ID: {hotel_id}")
        return pd.DataFrame()  # Return an empty DataFrame if no match
    idx = matching_indices[0]

    # Get the pairwise similarity scores of all hotels with that hotel
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the hotels based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the nums most similar hotels (Ignoring the hotel itself)
    sim_scores = sim_scores[1:nums+1]

    # Get the hotel indices
    hotel_indices = [i[0] for i in sim_scores]

    # Return the top n most similar hotels as a DataFrame
    return df.iloc[hotel_indices]

# Hiển thị đề xuất ra bảng
def display_recommended_hotels(recommended_hotels, cols=5):
    for i in range(0, len(recommended_hotels), cols):
        cols = st.columns(cols)
        for j, col in enumerate(cols):
            if i + j < len(recommended_hotels):
                hotel = recommended_hotels.iloc[i + j]
                with col:   
                    st.write(hotel['Hotel_Name'])                    
                    expander = st.expander(f"Description")
                    hotel_description = hotel['Hotel_Description']
                    truncated_description = ' '.join(hotel_description.split()[:100]) + '...'
                    expander.write(truncated_description)
                    expander.markdown("Nhấn vào mũi tên để đóng hộp text này.")           
def surprise_Recommender(New_ID, date, Model, num):
    New_ID_idx = date.loc[date['New_ID'] == New_ID, 'New_ID_idx'].iloc[0] 
    df_score = date[["Hotel_ID_idx", 'Hotel_ID']]
    df_score['EstimateScore'] = df_score['Hotel_ID_idx'].apply(lambda x: Model.predict(New_ID_idx, x).est) # est: get EstimateScore
    df_score = df_score.sort_values(by=['EstimateScore'], ascending=False)
    df_score = df_score.drop_duplicates()
    df_score = df_score[(df_score['EstimateScore'] >= 9.0)]
    df_info = pd.read_csv("hotel_info_VI.csv")
    df_score = pd.merge(df_score, df_info, on='Hotel_ID', how='inner')
    return df_score[['Hotel_ID', 'Hotel_Name', 'Hotel_Address']].head(num)
# Đọc dữ liệu khách sạn
df_hotels = pd.read_csv('hotel_info_VI.csv')
df_hotels_comments = pd.read_csv('hotel_comments_ID_Encoder.csv')

st.title("Data Science Project")
st.write("## Recommender System")
menu = ["Business Objective", "Build Project", "Content-based prediction", "Collaborative Prediction"]
choice = st.sidebar.selectbox('Menu', menu)
if choice == 'Business Objective':    
    st.subheader("Business Objective")
    st.write("""
    ###### Classifying spam and ham messages is one of the most common natural language processing tasks for emails and chat engines. With the advancements in machine learning and natural language processing techniques, it is now possible to separate spam messages from ham messages with a high degree of accuracy.
    """)  
    st.write("""###### => Problem/ Requirement: Use Machine Learning algorithms in Python for ham and spam message classification.""")
    st.image("hotel.jpg")

elif choice == 'Build Project':
    st.subheader("Build Project")
elif choice == 'Content-based prediction':
    st.subheader("Content-based prediction")
    # Lấy 10 khách sạn
    random_hotels = df_hotels.head(n=10)
    # print(random_hotels)

    st.session_state.random_hotels = random_hotels

    # Open and read file to cosine_sim_new
    with open('cosine_sim.pkl', 'rb') as f:
        cosine_sim_new = pickle.load(f)

    ###### Giao diện Streamlit ######
    st.image('hotel.jpg', use_column_width=True)

    # Kiểm tra xem 'selected_hotel_id' đã có trong session_state hay chưa
    if 'selected_hotel_id' not in st.session_state:
        # Nếu chưa có, thiết lập giá trị mặc định là None hoặc ID khách sạn đầu tiên
        st.session_state.selected_hotel_id = None

    # Theo cách cho người dùng chọn khách sạn từ dropdown
    # Tạo một tuple cho mỗi khách sạn, trong đó phần tử đầu là tên và phần tử thứ hai là ID
    hotel_options = [(row['Hotel_Name'], row['Hotel_ID']) for index, row in st.session_state.random_hotels.iterrows()]
    st.session_state.random_hotels
    # Tạo một dropdown với options là các tuple này
    selected_hotel = st.selectbox(
        "Chọn khách sạn",
        options=hotel_options,
        format_func=lambda x: x[0]  # Hiển thị tên khách sạn
    )
    # Display the selected hotel
    st.write("Bạn đã chọn:", selected_hotel)

    # Cập nhật session_state dựa trên lựa chọn hiện tại
    st.session_state.selected_hotel_id = selected_hotel[1]

    if st.session_state.selected_hotel_id:
        st.write("Hotel_ID: ", st.session_state.selected_hotel_id)
        # Hiển thị thông tin khách sạn được chọn
        selected_hotel = df_hotels[df_hotels['Hotel_ID'] == st.session_state.selected_hotel_id]

        if not selected_hotel.empty:
            st.write('#### Bạn vừa chọn:')
            st.write('### ', selected_hotel['Hotel_Name'].values[0])

            hotel_description = selected_hotel['Hotel_Description'].values[0]
            truncated_description = ' '.join(hotel_description.split()[:100])
            st.write('##### Thông tin:')
            st.write(truncated_description, '...')

            st.write('##### Các khách sạn khác bạn cũng có thể quan tâm:')
            recommendations = get_recommendations(df_hotels, st.session_state.selected_hotel_id, cosine_sim=cosine_sim_new, nums=3) 
            display_recommended_hotels(recommendations, cols=3)
        else:
            st.write(f"Không tìm thấy khách sạn với ID: {st.session_state.selected_hotel_id}")
elif choice == 'Collaborative Prediction':
    st.subheader("Collaborative Prediction")
    # Lấy khách hàng
    random_hotels = df_hotels_comments.head(n=10)
    # print(random_hotels)

    st.session_state.random_hotels = random_hotels
    # print(random_hotels)

    # Open and read file to cosine_sim_new
    with open('SVD_Surprise.pkl', 'rb') as f:
        cosine_sim_new1 = pickle.load(f)

   
    # Kiểm tra xem 'selected_hotel_id' đã có trong session_state hay chưa
    if 'selected_hotel_id' not in st.session_state:
        # Nếu chưa có, thiết lập giá trị mặc định là None hoặc ID khách sạn đầu tiên
        st.session_state.selected_hotel_id = None

    # Theo cách cho người dùng chọn khách sạn từ dropdown
    # Tạo một tuple cho mỗi khách sạn, trong đó phần tử đầu là tên và phần tử thứ hai là ID
    hotel_options = [(row['Reviewer_Name'], row['New_ID']) for index, row in st.session_state.random_hotels.iterrows()]
    st.session_state.random_hotels
    # Tạo một dropdown với options là các tuple này
    selected_hotel = st.selectbox(
        "Chọn khách sạn",
        options=hotel_options,
        format_func=lambda x: x[1]  # Hiển thị tên khách sạn
    )
    # Display the selected hotel
    st.write("Bạn đã chọn:", selected_hotel)

    # Cập nhật session_state dựa trên lựa chọn hiện tại
    st.session_state.selected_hotel_id = selected_hotel[1]

    if st.session_state.selected_hotel_id:
        st.write("Hotel_ID: ", st.session_state.selected_hotel_id)
        # Hiển thị thông tin khách sạn được chọn
        selected_hotel = df_hotels_comments[df_hotels_comments['New_ID'] == st.session_state.selected_hotel_id]

        if not selected_hotel.empty:
            st.write('#### Bạn vừa chọn:')
            st.write('### ', selected_hotel['Reviewer_Name'].values[0])

            hotel_description = selected_hotel['Nationality'].values[0]
            truncated_description = ' '.join(hotel_description.split()[:100])
            st.write('##### Thông tin:')
            # st.write(truncated_description, '...')

            st.write('##### Các khách sạn khác bạn cũng có thể quan tâm:')
            recommendations = surprise_Recommender(st.session_state.selected_hotel_id, df_hotels_comments, cosine_sim_new1, 3) 
            display_recommended_hotels(recommendations, cols=3)
        else:
            st.write(f"Không tìm thấy khách sạn với ID: {st.session_state.selected_hotel_id}")
