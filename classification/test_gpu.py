import tensorflow as tf

# Lấy danh sách các GPU vật lý mà TensorFlow nhìn thấy
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print(f"Ngon lành! TensorFlow đang nhận diện được {len(gpus)} GPU:")
    for gpu in gpus:
        print(" -", gpu)
else:
    print("Báo động: TensorFlow không tìm thấy GPU nào, đang chạy bằng CPU nha bro! (Nhớ check lại driver, CUDA hoặc cuDNN)")
import tensorflow as tf

build_info = tf.sysconfig.get_build_info()
print("Phiên bản CUDA TensorFlow đang nhận:", build_info.get('cuda_version', 'Không tìm thấy'))
print("Phiên bản cuDNN TensorFlow đang nhận:", build_info.get('cudnn_version', 'Không tìm thấy'))