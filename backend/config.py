import multiprocessing

num_cores = multiprocessing.cpu_count()
allow_img_extensions = ('.jpg', '.png', '.jpeg', '.gif')  # in lower format
stop_words_path = './backend/stopwords.txt'
history_meme_path = './history/'
