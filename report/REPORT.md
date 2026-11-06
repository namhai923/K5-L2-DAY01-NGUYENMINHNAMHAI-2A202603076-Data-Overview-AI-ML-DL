# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=… ; `class_name`=… ; `rank`=1 ; `score`=… ; `taxonomy_name`=…
  ```
  {
    "sample_id": "traffic",
    "coco_image_id": 210273,
    "image_width": 640,
    "image_height": 428,
    "task": "image_classification",
    "taxonomy_name": "ImageNet-1K",
    "model_file": "yolo11n-cls.pt",
    "model_sha256": "c62d41bf9625777760018bf914d2e6cd472420ccd01706d97a61cb6c82502bd7",
    "ultralytics_version": "8.4.145",
    "rank": 1,
    "class_id": 468,
    "class_name": "cab",
    "score": 0.510915
  },
  ```

2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
    Về cả bức ảnh, vì tôi thấy các attribute trả về từ prediction liên quan nhiều về phân loại ảnh như `image_id`, `image_width`, `image_height` và tên của task cũng là `image_classification`

3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
    Danh sách lớp đi kèm taxonomy/checkpoint

4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
    Vì mỗi taxonomy sẽ có class_id và class_name tương ứng, nên 2 taxonomy khác nhau sẽ có class id/name khác nhau

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
    Guideline nên có quy tắc chọn một lớp chính:
    - Chọn lớp chiếm diện tích lớn nhất hoặc nổi bật nhất và là trọng tâm của ảnh
    - Nếu nhiều vật ngang nhau, xét theo thứ tự ưu tiên lớp do guideline quy định

6. Vì sao không được lấy `score` của model làm ground truth?
    Vì score chỉ là mức tin của model vào dự đoán của chính nó

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=… ; `score`=… ; `bbox_xyxy`=[…, …, …, …] ; `bbox_width`=… ; `bbox_height`=…
  ```
  {
    "sample_id": "traffic",
    "coco_image_id": 210273,
    "image_width": 640,
    "image_height": 428,
    "task": "object_detection",
    "taxonomy_name": "COCO-80",
    "model_file": "yolo11n.pt",
    "model_sha256": "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
    "ultralytics_version": "8.4.145",
    "score_threshold": 0.35,
    "class_id": 5,
    "class_name": "bus",
    "score": 0.912558,
    "coordinate_unit": "pixel",
    "bbox_format": "xyxy",
    "bbox_xyxy": [
      93.17,
      187.95,
      223.01,
      320.91
    ],
    "bbox_width": 129.84,
    "bbox_height": 132.96
  },
  ```
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
    - Góc bên trên-trái: (93.17,187.95)
    - Góc bên dưới-phải: (223.01,320.91)
    - Rộng: 129.84
    - cao: 132.96


3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
    - threshold=0.20: 17 predictions 
    - threshold=0.60: 6 predictions

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
    - đối chiếu với ảnh và guideline, không tin theo score
    - kiểm tra lại chất lượng các box score thấp
    - loại false positive, ví dụ vật bị gán nhầm lớp
    - xử lý box trùng hoặc chồng nhau trên cùng một vật

5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
    - Mỗi vật thuộc lớp trong danh sách được vẽ đúng một box, ôm sát toàn bộ phần nhìn thấy của vật, không chừa nền thừa và không cắt mất phần nhìn thấy. Không vẽ phần bị che theo phỏng đoán. Nếu vật bị cắt ở mép ảnh, box dừng tại mép ảnh

6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
    - Vật được chọn là `oven 0.69` ở bên trái ảnh: phần dưới bị cái bàn phía trước che, và cạnh trái chạm mép ảnh.  Tôi viết ca này thành quy tắc trong guideline, không escalation vì phần nhìn thấy đủ để xác định có đúng là `oven` hay không
    

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=… ; `class_name`=… ; `score`=… ; số điểm polygon=`polygon_point_count`=… ; 3–5 điểm đầu của `polygon_xy`=…
    ```
    {
    "sample_id": "traffic",
    "coco_image_id": 210273,
    "image_width": 640,
    "image_height": 428,
    "task": "instance_segmentation",
    "taxonomy_name": "COCO-80",
    "model_file": "yolo11n-seg.pt",
    "model_sha256": "55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152",
    "ultralytics_version": "8.4.145",
    "score_threshold": 0.35,
    "instance_id": "traffic-001",
    "class_id": 5,
    "class_name": "bus",
    "score": 0.925745,
    "coordinate_unit": "pixel",
    "bbox_format": "xyxy",
    "bbox_xyxy": [
      95.3,
      188.72,
      224.08,
      319.96
    ],
    "polygon_point_count": 120,
    "polygon_xy": [
      [
        148.0,
        189.0
      ],
      [
        147.0,
        190.0
      ],
      [
        145.0,
        190.0
      ],
      [
        143.0,
        192.0
      ],
      [
        142.0,
        192.0
      ],
      ...
    ```
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
    Box chỉ cho biết vị trí và kích thước bằng một hình chữ nhật, nên chứa cả nền xung quanh vật. Polygon gồm các điểm bám theo biên nên cho biết hình dạng thật của vật

3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
    Nó cho biết từng vật thể riêng biệt. 2 cái bát có thể cùng class `bowl` nhưng có thể có `instance_id` là `bowl-01` và `bowl-02`. Nó cũng ko phải `tracking_id` vì nó không phải cùng một vật thể di chuyển ở 2 thời điểm khác nhau
    
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
    Vẽ mỗi instance một mask, biên bám sát mép phần nhìn thấy của vật. Không tô nền, không tô vật khác nằm trên hoặc che phía trước, và không tô phần bị che theo phỏng đoán. Tại chỗ hai vật chạm nhau, mỗi pixel chỉ thuộc một instance


5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
    Tôi chọn `dining table 0.58`. Tôi dùng guideline vì vật đặt trên hoặc che bàn sẽ lặp lại ở nhiều ảnh và ở đây nhìn rõ đâu là bàn, đâu là vật khác

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một lớp cho **cả ảnh**, do người gán theo guideline: `class_id`, `class_name`, `taxonomy_name`. | `traffic`: top-1 là `cab` (`score` 0.510915) nhưng xe buýt nổi bật hơn và không thấy rõ dấu hiệu taxi. Ảnh có nhiều loại xe, top-5 còn có `minibus`, `streetcar`. | Xác định chủ thể chính theo guideline rồi chọn một lớp trong taxonomy. Escalation nếu không chọn được lớp phù hợp. | Lớp có đúng guideline và taxonomy không; các lớp dễ nhầm có được xử lý nhất quán không. Không dùng `score` để kết luận nhãn đúng; nhãn sai thì trả lại để rework. |
| Phát hiện vật thể | Một lớp và một box `bbox_xyxy` (pixel) cho **mỗi vật**. | `kitchen`: `oven` 0.69 bị bàn che nhưng box kéo xuống cả mặt bàn. `bowl` 0.38, `bowl` 0.50 và `cup` 0.45 chồng lấn nhau. Số prediction đổi theo ngưỡng: 17 ở 0.20, 6 ở 0.60. | Vẽ mỗi vật một box, ôm sát phần nhìn thấy, không phỏng đoán phần bị che. Gán vật theo guideline, không bỏ vật vì score thấp. Escalation nếu không xác định được vật. | Có sót vật, thừa box hoặc trùng box không; lớp có đúng không; box có ôm sát phần nhìn thấy không. Đối chiếu với ảnh, không lọc theo threshold; lỗi thì trả lại để rework. |
| Instance segmentation | Một lớp, một `instance_id` và một mask/polygon `polygon_xy` (pixel) cho **mỗi vật**. | `kitchen`: mask `dining table` 0.58 tô lên bột, cây cán bột và tràn sang thùng kim loại dưới bàn, trong khi lọ trên bàn được tách ra. | Vẽ mỗi instance một mask bám sát biên phần nhìn thấy, không tô vật khác. Tách riêng các vật chạm nhau. Escalation nếu biên quá mờ hoặc tối, không xác định được. | Biên có sát không; mask có tràn sang vật khác hoặc thiếu phần nhìn thấy không; mỗi vật có đúng một instance không; hai instance có tô chung pixel không. Lỗi thì trả lại để rework. |


## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
    Chỉ dùng các ảnh mẫu công khai có sẵn trong notebook, không tải lên ảnh riêng, dữ liệu khách hàng/VinFast hoặc dữ liệu cá nhân, và không ghi họ tên, MSSV, email hay số điện thoại vào báo cáo/output
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
    Tôi sẽ dừng ngay, không xử lý, không chia sẻ và không đưa dữ liệu đó lên repo, rồi báo cho Lab Coach/GV để được hướng dẫn

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
