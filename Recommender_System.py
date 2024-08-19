import streamlit as st
import numpy as np
import pandas as pd
import time
import pickle
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from surprise import Reader, Dataset
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate
from surprise import SVD
from surprise import accuracy
import io
from wordcloud import WordCloud

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
    return df_score.head(num)
def plot_boxplots(results_df):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.boxplot(data=results_df[['Train RMSE', 'Test RMSE']], palette="Set2", ax=axs[0])
    axs[0].set_title('RMSE for Train and Test Sets')
    axs[0].set_ylabel('RMSE')
    
    sns.boxplot(data=results_df[['Train MAE', 'Test MAE']], palette="Set2", ax=axs[1])
    axs[1].set_title('MAE for Train and Test Sets')
    axs[1].set_ylabel('MAE')
    
    return fig
def plot_barplot(results_df):
    avg_results = results_df.mean(axis=0)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=avg_results.index, y=avg_results.values, palette="Blues_d", ax=ax)
    ax.set_title('Average Cross-Validation Results')
    ax.set_ylabel('Average Score')
    return fig

# Đọc dữ liệu khách sạn
df_hotels = pd.read_csv('hotel_info_VI.csv')
df_hotels_comments = pd.read_csv('hotel_comments_ID_Encoder.csv')

def plot_cosine_similarity_matrix(cosine_sim, labels):
    # Tạo một đối tượng BytesIO để lưu biểu đồ
    buf = io.BytesIO()
    
    # Tạo biểu đồ
    plt.figure(figsize=(12, 10))
    sns.heatmap(cosine_sim, cmap='viridis', annot=False, fmt=".2f", cbar_kws={'shrink': .8})
    plt.title("Cosine Similarity Matrix")
    plt.xlabel("Document Index")
    plt.ylabel("Document Index")
    
    # Thiết lập các nhãn trục x và y với khoảng cách
    step = 100  # Điều chỉnh giá trị này để phù hợp với số lượng tài liệu
    plt.xticks(ticks=np.arange(0, len(labels), step=step), labels=np.arange(0, len(labels), step=step), rotation=90)
    plt.yticks(ticks=np.arange(0, len(labels), step=step), labels=np.arange(0, len(labels), step=step), rotation=0)
    
    # Lưu biểu đồ vào đối tượng BytesIO
    plt.savefig(buf, format="png")
    buf.seek(0)  # Đưa con trỏ về đầu đối tượng
    plt.close()  # Đóng biểu đồ để giải phóng tài nguyên
    
    return buf
def plot_ward_distribution(ward_counts):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=ward_counts, x='Phuong', y='Number of Hotels', palette='viridis')
    plt.title('Number of Hotels per Ward')
    plt.xlabel('Ward')
    plt.ylabel('Number of Hotels')
    plt.xticks(rotation=65)
    plt.tight_layout()
    st.pyplot(plt)
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
    st.write("## Content-based")
    with open('cosine_sim.pkl', 'rb') as f:
        cosine_sim_new = pickle.load(f)
    st.write("##### 1. Some data")
    st.dataframe(df_hotels[['Hotel_ID', 'Phuong', 'Content_wt']].head(3),height=300, use_container_width=True)
    st.dataframe(df_hotels[['Hotel_ID', 'Phuong', 'Content_wt']].tail(3), use_container_width=True)  
    st.write("##### 2. Visualize Content")
    
    st.subheader('Content Word Cloud')
    content_text = ' '.join(df_hotels['Content_wt'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(content_text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)
    # Hiển thị biểu đồ trong Streamlit
    ward_counts = df_hotels['Phuong'].value_counts().reset_index()
    ward_counts.columns = ['Phuong', 'Number of Hotels']
    st.title('Hotel Distribution by Ward')
    # Tạo và hiển thị biểu đồ
    plot_ward_distribution(ward_counts)
    st.write("##### 3. Build model...")
    st.write("##### 4. Evaluation")
    
    st.subheader("Cosine Similarity Matrix")
    buf = plot_cosine_similarity_matrix(cosine_sim_new, df_hotels)
    st.image(buf, caption="Cosine Similarity Matrix", use_column_width=True)
    
    start_time = time.time()
    recommendations = get_recommendations(df_hotels, '1_1', cosine_sim=cosine_sim_new, nums=3) 
    print("Thời gian chạy Cosine: %s seconds" % (time.time() - start_time))
    st.write("Thời gian chạy Cosine:",(time.time() - start_time))
    
    st.write("## Collaborative")
    reader = Reader(rating_scale=(0, 10))
    data = Dataset.load_from_df(df_hotels_comments[["New_ID_idx","Hotel_ID_idx", "Score"]], reader)
    with open('SVD_Surprise.pkl', 'rb') as f:
        SVD_Surprise = pickle.load(f)
        
    st.write("##### 1. Some data")
    st.dataframe(df_hotels_comments.head(3))
    st.dataframe(df_hotels_comments.tail(3))  
    st.write("##### 2. Visualize Ham and Spam")
   
    st.write("##### 3. Build model...")
    st.write("##### 4. Evaluation") 
    
    results = cross_validate(SVD_Surprise, data, measures=['RMSE', 'MAE'], cv=5, return_train_measures=True, verbose=True)
    st.dataframe(pd.DataFrame.from_dict(results).mean(axis=0))
    results_df = pd.DataFrame.from_dict(results)
    results_df = results_df[['train_rmse', 'test_rmse', 'train_mae', 'test_mae']]
    results_df.columns = ['Train RMSE', 'Test RMSE', 'Train MAE', 'Test MAE']
    st.title("Evaluation of SVD Model using Surprise")

    # Hiển thị kết quả cross-validation dưới dạng bảng
    st.subheader("Cross-Validation Results")
    st.write(results_df)
    
    # Hiển thị biểu đồ Boxplot
    st.subheader("Boxplot of RMSE and MAE")
    boxplot_fig = plot_boxplots(results_df)
    st.pyplot(boxplot_fig)
    
    # Hiển thị biểu đồ Barplot
    st.subheader("Average Cross-Validation Scores")
    barplot_fig = plot_barplot(results_df)
    st.pyplot(barplot_fig)

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
    # Lấy người dùng
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

    # Theo cách cho người dùng đăng nhập dropdown
    # Tạo một tuple cho mỗi người dùng, trong đó phần tử đầu là tên và phần tử thứ hai là ID
    hotel_options = [(row['Reviewer_Name'], row['New_ID']) for index, row in st.session_state.random_hotels.iterrows()]
    st.session_state.random_hotels
    # Tạo một dropdown với options là các tuple này
    selected_hotel = st.selectbox(
        "Đăng nhập theo New_ID ",
        options=hotel_options,
        format_func=lambda x: x[1] 
    )
    # Display the selected hotel
    st.write("Bạn đã chọn:", selected_hotel)

    # Cập nhật session_state dựa trên lựa chọn hiện tại
    st.session_state.selected_hotel_id = selected_hotel[1]

    if st.session_state.selected_hotel_id:
        st.write("New_ID: ", st.session_state.selected_hotel_id)
        # Hiển thị thông tin khách sạn được chọn
        selected_hotel = df_hotels_comments[df_hotels_comments['New_ID'] == st.session_state.selected_hotel_id]

        if not selected_hotel.empty:
            st.write('#### Bạn đăng nhập với tên là :', selected_hotel['Reviewer_Name'].values[0])
            name =  selected_hotel['Reviewer_Name'].values[0];
            Nationality = selected_hotel['Nationality'].values[0]
            st.write('##### Thông tin:')
            st.write('Tên:', name, ', Quốc gia:', Nationality, '...')

            st.write('##### Các khách sạn khác bạn cũng có thể quan tâm:')
            recommendations = surprise_Recommender(st.session_state.selected_hotel_id, df_hotels_comments, cosine_sim_new1, 3) 
            display_recommended_hotels(recommendations, cols=3)
        else:
            st.write(f"Không tìm thấy khách sạn với ID: {st.session_state.selected_hotel_id}")
