Project
======================

* Bài toán phân loại ảnh (~400.000) gồm nhiều class (~5000)
* Data có nhiều ảnh nhiễu, không đúng ảnh người nổi tiếng
* Dùng thư viện face_recognition để tính vector embedding (128 chiều) cho mỗi ảnh trong dataset
	* Các ảnh của cùng 1 người sẽ có vector embedding gần nhau
	* Với mỗi mid (người) chọn ra top những ảnh có *mức độ giống* với các ảnh còn lại trong folder nhất để giữ lại. Loại các ảnh không giống với các ảnh khác trong folder, vì đây khả năng cao là các ảnh nhiễu.
	* Mục đích: Loại ảnh nhiễu khỏi tập train

* Train mô hình Machine Learning
	* Với mỗi ảnh thì sử dụng feature là vector embedding tính bởi thư viện face_recognition
	* Train các mô hình SVM, Random Forest, KNN
		* Thời gian train khá lâu, chỉ test được độ chính xác của KNN (validation accuracy: ~80%)

* Train mô hình mạng neuron
	* Chỉ dùng một tập nhỏ data để train, validation (~28k ảnh, 500 người và mỗi người có ít nhất 50 ảnh)
	* Thử nghiệm các mô hình pretrained VGG16, Resnet50, tự code mô hình VGG thu nhỏ
		* Train 100 epoch (khoảng 4-5h)
		* Learning rate cần phải nhỏ, xấp xỉ khoảng 2*10e-4 (nếu lr=10e-3 thì valid accuracy chỉ ở mức 0.25% ~ mô hình đoán ngẫu nhiên)
		* Mọi mô hình đều overfit nặng (92% train accuracy - 72% valid accuracy, nếu train thêm thì train accuracy tăng đến 96% còn valid accuracy không tăng)
	* Thử train mạng Siamese với base network là mô hình đã train ở bước trước, nhưng kết quả không cải thiện (có thể do code sai)