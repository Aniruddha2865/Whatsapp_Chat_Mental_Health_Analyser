from urlextract import URLExtract
extract = URLExtract()
from wordcloud import WordCloud
import emoji
from collections import Counter
import pandas as pd

def fetch_stats(selected_user, df):
    if selected_user != "Overall":
        df = df[df['users'] == selected_user]

    #1. no. of msgs
    num_msgs = df.shape[0]

    #2. no. of words
    words = []
    for i in df["msgs"]:
        words.extend(i.split())

    #3. no. of media msgs
    num_media = df[df["msgs"] == "<Media omitted>"].shape[0]

    #4. no. of links shared
    links = []
    for i in df["msgs"]:
        links.extend(extract.find_urls(i))

    return num_msgs, len(words), num_media, len(links)

def fetch_top_users(df):
    #top users
    x =  df[df["users"] != "group_notification"]["users"].value_counts().head()
    
    # %age of msgs per user
    df = round((df[df["users"] != "group_notification"]["users"].value_counts()/df.shape[0])*100 ,2).reset_index().rename(columns = {"count":"percent"})
    return x, df

def emoji_helper(selected_user, df):
    if selected_user != "Overall":
        df = df[df['users'] == selected_user] 

    emojis = []
    for i in df["msgs"]:
        emojis.extend([c for c in i if emoji.is_emoji(c)])
    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis)))).rename(columns={0: 'Emoji', 1: 'Count'})
    return emoji_df

def wordcloud(selected_user,df):

    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['users'] == selected_user]

    temp = df[df['users'] != 'group_notification']
    unwanted = {'media', 'omitted'}

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            word = word.strip("<>.,!?")   # remove symbols

            if word not in stop_words and word not in unwanted and word != "":
                y.append(word)

        return " ".join(y)
    
    if temp.empty:
        return None
    
    temp['msgs'] = temp['msgs'].apply(remove_stop_words)

    text = temp['msgs'].str.cat(sep=" ").strip()

    if text == "":
        return None
    wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')
    return wc.generate(text)

def most_common_words(selected_user,df):

    f = open('stop_hinglish.txt','r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['users'] == selected_user]

    temp = df[df['users'] != 'group_notification']

    words = []
    unwanted = {'media', 'omitted'}

    for message in temp['msgs']:
        for word in message.lower().split():
            word = word.strip("<>.,!?")   # remove symbols

            if word not in stop_words and word not in unwanted and word != "":
                words.append(word)

    if len(words) == 0:
        return pd.DataFrame(columns=[0, 1])

    most_common_df = pd.DataFrame(Counter(words).most_common(15))
    return most_common_df

def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['users'] == selected_user]

    timeline = df.groupby(["year", "month_num", "month"]).count()["msgs"].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline["month"][i] + "-" + str(timeline["year"][i]))

    timeline["time"] = time
    return timeline

def daily_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['users'] == selected_user]

    daily_timeline = df.groupby(["only_date"]).count()["msgs"].reset_index()
    return daily_timeline




def get_mental_df(selected_user, df, tokenizer, model, predict_func):
    temp_df = df.copy()

    # remove system messages
    temp_df = temp_df[temp_df["users"] != "group_notification"]

    # apply user filter
    if selected_user != "Overall":
        temp_df = temp_df[temp_df["users"] == selected_user]

    # limit for speed
    temp_df = temp_df.head(200)

    predictions = []

    for msg in temp_df["msgs"]:
        try:
            pred = predict_func(msg, tokenizer, model)
        except:
            pred = -1
        predictions.append(pred)

    temp_df["mental_state"] = predictions

    label_map = {
        0: "anxiety",
        1: "depression",
        2: "mental_disorder",
        3: "normal",
        4: "suicidewatch"
    }

    temp_df["mental_label"] = temp_df["mental_state"].map(label_map)

    return temp_df

def get_mental_distribution(temp_df):
    return temp_df["mental_label"].value_counts()

def get_user_mental_table(temp_df, selected_user):
    all_labels = ["anxiety", "depression", "mental_disorder", "normal", "suicidewatch"]

    if selected_user == "Overall":
        user_mental = (
            temp_df.groupby(["users", "mental_label"])
            .size()
            .unstack()
            .reindex(columns=all_labels, fill_value=0)
            .fillna(0)
        )
    else:
        user_mental = (
            temp_df["mental_label"]
            .value_counts()
            .to_frame(name="count")
            .T
            .reindex(columns=all_labels, fill_value=0)
            .fillna(0)
        )

    return user_mental

