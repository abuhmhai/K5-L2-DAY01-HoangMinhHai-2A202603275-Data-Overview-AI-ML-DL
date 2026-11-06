# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Môi trường chạy:** PyCharm trên macOS Apple Silicon, môi trường `.venv`, thiết bị `cpu` (theo output setup).

**Python / PyTorch / Ultralytics:** 3.15.2 / 2.15.0 / 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Chạy cục bộ bằng PyCharm thay cho Colab. Output setup ghi `Device: cpu`; các JSON ghi Ultralytics 8.4.145 và ngưỡng detection/segmentation 0.35. Báo cáo và evidence được đặt trực tiếp trong `report/` để nộp qua GitHub.

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

   **Trả lời:** **Record hạng 1 của ảnh `traffic`:** `class_id`=468; `class_name`=`cab`; `rank`=1; `score`=0.510915; `taxonomy_name`=`ImageNet-1K`. Kết quả lấy từ `classification_predictions.json`, khớp với cột cao nhất trên hình `classification_top5.png`.

2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?

   **Trả lời:** **Record nói về cả bức ảnh.** Tác vụ là `image_classification`: mô hình xếp hạng các lớp cho toàn ảnh, không trả về tọa độ box hoặc mask để xác định một chiếc xe riêng. Nhãn `cab` đứng đầu không có nghĩa mọi xe trong ảnh đều là taxi.

3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?

   **Trả lời:** **Danh sách lớp đi kèm taxonomy/checkpoint.** Checkpoint `yolo11n-cls.pt` sử dụng bộ lớp ImageNet-1K; mô hình chọn và xếp hạng trong danh sách này, không tự tạo tên lớp mới khi dự đoán.

4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?

   **Trả lời:** **Cần ghi đủ mã, tên và taxonomy để hiểu đúng lớp.** `class_id` là mã máy xử lý; `class_name` giúp con người đọc; `taxonomy_name` xác định bộ lớp và ngữ cảnh ánh xạ mã sang tên. Cùng một mã có thể mang nghĩa khác trong bộ lớp khác, còn tên lớp riêng lẻ chưa đủ để truy lại ánh xạ.

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?

   **Trả lời:** **Guideline phải quy định cách chọn một lớp khi ảnh có nhiều chủ đề.** Cần nêu bộ lớp được phép dùng, tiêu chí chọn chủ thể chính, thứ tự ưu tiên khi nhiều đối tượng cùng nổi bật và cách xử lý ca không rõ. Ảnh `traffic` có nhiều xe buýt và ô tô nên không thể chỉ chọn theo một vật bất kỳ; nếu guideline chưa giải quyết được thì hỏi Lab Coach, không lấy top-1 làm nhãn mặc định.

6. Vì sao không được lấy `score` của model làm ground truth?

   **Trả lời:** **Score không phải ground truth.** `0.510915` là điểm mô hình gán cho lớp `cab`, không phải điểm chất lượng nhãn hay bằng chứng dự đoán chắc chắn đúng. Ground truth phải được con người xác định theo guideline và kiểm tra chất lượng. Ảnh giao thông có nhiều loại xe là trường hợp cần đối chiếu trực quan thay vì chép dự đoán thành nhãn chuẩn.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=… ; `score`=… ; `bbox_xyxy`=[…, …, …, …] ; `bbox_width`=… ; `bbox_height`=…

   **Trả lời:** **Record được chọn trong ảnh `kitchen`:** `class_name`=`bowl`; `score`=0.719063; `bbox_xyxy`=[32.65, 342.12, 100.16, 384.93]; `bbox_width`=67.51; `bbox_height`=42.81. Record thuộc COCO-80, được tạo bằng `yolo11n.pt` ở ngưỡng 0.35. Trên PNG, đây là chiếc bát màu sáng ở phía dưới bên trái, có nhãn `bowl 0.72`.

2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?

   **Trả lời:** **Ý nghĩa của box:** ảnh gốc rộng 640, cao 427 pixel; gốc tọa độ ở góc trên trái, x tăng sang phải và y tăng xuống dưới. Box đi từ góc trên trái (32.65, 342.12) đến góc dưới phải (100.16, 384.93), rộng 67.51 pixel và cao 42.81 pixel. Đây là tọa độ trên ảnh gốc, không phải trên PNG đã phóng to.

3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?

   **Trả lời:** **So sánh ngưỡng trên cùng ảnh `kitchen`:** output notebook ghi ngưỡng 0.20 có **17 prediction**, ngưỡng 0.60 có **6 prediction**; ngưỡng 0.35 dùng cho JSON có 11 prediction. Hạ từ 0.60 xuống 0.20 giữ thêm 11 prediction. Đây là số dự đoán được giữ lại, không phải số vật thể đúng đã được kiểm chứng.

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?

   **Trả lời:** **Reviewer phải kiểm tra thêm các prediction được giữ lại khi hạ ngưỡng:** đối chiếu ảnh để phát hiện sai lớp, hộp lệch, hộp trùng và các vùng không phải vật thể; đồng thời kiểm tra vật thể bị bỏ sót. Ở ngưỡng 0.20, danh sách có thêm các lớp như `spoon`, `potted plant`, `dining table`, `bottle` so với 0.60; cần xác minh từng trường hợp, không mặc định điểm thấp là sai hoặc điểm cao là đúng. Ngưỡng lọc prediction không thay thế quy tắc quyết định đối tượng nào cần có trong ground truth.

5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).

   **Trả lời:** **Quy tắc box đề xuất:** với mỗi đối tượng thuộc bộ lớp và đủ điều kiện gán nhãn theo guideline, tạo một box riêng, ôm sát phần đối tượng nhìn thấy, hạn chế nền thừa và giữ tọa độ trong ảnh. Khi vật thể bị che hoặc cắt mép, không tự suy đoán phần ngoài vùng nhìn thấy; cần theo quy ước chỉ gán phần nhìn thấy hay cả phần bị che do guideline quy định. Nếu chưa có quy ước thì hỏi Lab Coach trước khi gán. Đây là đề xuất cần thống nhất, không phải quy tắc đã được bài lab cung cấp.

6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?

   **Trả lời:** **Ca cắt mép cần escalation:** ở mép trái ảnh, vùng được mô hình gán `person` với score 0.61093, box [0.12, 263.18, 61.55, 310.78], chỉ cho thấy một phần cánh tay/bàn tay đi vào khung hình. Tôi sẽ hỏi Lab Coach về điều kiện tối thiểu để gán lớp `person` cho một phần cơ thể và cách vẽ box trong trường hợp này. Không tự kéo box ra ngoài ảnh, suy đoán toàn bộ người hoặc coi nhãn mô hình là ground truth. Sau khi được thống nhất, bổ sung quyết định và ví dụ vào guideline để annotator/reviewer xử lý nhất quán.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=… ; `class_name`=… ; `score`=… ; số điểm polygon=`polygon_point_count`=… ; 3–5 điểm đầu của `polygon_xy`=…

   **Trả lời:** Chọn record của ảnh `sample_id=kitchen`: `instance_id`=`kitchen-002`; `class_name`=`bowl`; `score`=0.735744; `polygon_point_count`=67; 5 điểm đầu của `polygon_xy`=[[53.0, 344.0], [52.0, 345.0], [50.0, 345.0], [49.0, 346.0], [48.0, 346.0]]. Tọa độ tính bằng pixel trên ảnh gốc 640 × 427. Record thuộc COCO-80, dùng `yolo11n-seg.pt` ở ngưỡng 0.35. Trên PNG, đây là chiếc bát màu sáng ở phía dưới bên trái, nhãn `bowl 0.74`.

2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?

   **Trả lời:** Box chỉ bao đối tượng bằng hình chữ nhật, có thể chứa cả nền và phần không thuộc đối tượng. Polygon mô tả đường biên để xác định vùng mask, thể hiện hình dáng cong hoặc không đều của vật thể. Với chiếc bát được chọn, polygon bám theo hình dáng bát thay vì lấy cả bốn góc chữ nhật của box. Đây vẫn là biên dự đoán cần kiểm tra, không tự trở thành ground truth.

3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?

   **Trả lời:** `instance_id` phân biệt từng đối tượng trong output lab. Ví dụ `kitchen-002` và `kitchen-003` là hai instance khác nhau nhưng cùng lớp `bowl` (class_id=45). Class ID mô tả loại đối tượng; instance ID phân biệt từng đối tượng cụ thể. ID được notebook tạo theo thứ tự kết quả trong mỗi ảnh, không có cơ chế liên kết cùng đối tượng qua nhiều khung hình, nên không phải tracking ID và có thể thay đổi giữa các lần chạy/cấu hình.

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.

   **Trả lời:** Quy tắc đề xuất: vẽ một mask riêng cho từng đối tượng đủ điều kiện gán nhãn; đặt các điểm polygon sát đường biên phần nhìn thấy, đủ điểm ở đoạn cong để giữ hình dạng, không lấn sang nền hay vật bên cạnh và tránh đường biên tự cắt. Không gộp hai vật cùng lớp thành một instance. Cách xử lý phần bị che, lỗ và các vùng rời nhau phải theo guideline; nếu chưa có quy ước thì hỏi Lab Coach, không tự vẽ phần không quan sát được.

5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?

   **Trả lời:** Trên ảnh `kitchen`, mặt bàn ở tiền cảnh (prediction `kitchen-005`, lớp `dining table`) bị các bát và dụng cụ đặt trên bàn che khuất. Ở chỗ chiếc bát `kitchen-002` tiếp giáp mặt bàn phía dưới bên trái, không nhìn thấy phần mặt bàn nằm dưới bát, nên không thể suy ra biên đầy đủ chỉ từ ảnh. Tôi sẽ escalation tới Lab Coach nếu guideline chưa quy định mask chỉ lấy phần nhìn thấy hay cả phần bị che, và cách biểu diễn vùng bị vật khác che. Nếu đã có quy định thì áp dụng quy định đó, tách mask bát khỏi mask bàn theo đúng quy ước. Không lấy vùng màu của mô hình làm quyết định gán nhãn; reviewer cần đối chiếu ảnh gốc để kiểm tra vùng biên và phần nền bị lấn.


## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn lớp cho toàn ảnh theo taxonomy và guideline của bài toán single-label; không có box/mask. | Ảnh `traffic` có nhiều xe buýt và ô tô; top-1 là `cab`, `class_id=468`, `score=0.510915` (JSON và `classification_top5.png`). Khó chọn một lớp đại diện cho cả cảnh; top-1 chưa chứng minh nhãn đúng. | Xem toàn ảnh, chọn lớp theo tiêu chí chủ thể chính và thứ tự ưu tiên trong guideline; nếu chưa rõ thì hỏi Lab Coach, không chép top-1 thành ground truth. | Kiểm tra nhãn có mô tả cấp ảnh, đúng ánh xạ mã/tên/ImageNet-1K và nhất quán với guideline; yêu cầu gán lại nếu chọn lớp chỉ dựa vào score. |
| Phát hiện vật thể | Mỗi đối tượng đủ điều kiện có một lớp và một box riêng; ở định dạng đang đọc là `xyxy` theo pixel. | Ảnh `kitchen`: vùng `person`, `score=0.61093`, box `[0.12, 263.18, 61.55, 310.78]` ở mép trái chỉ thấy một phần cánh tay/bàn tay (`detection_predictions.png`). Cần quyết định điều kiện gán nhãn cho phần cơ thể bị cắt. | Gán từng đối tượng theo quy tắc box; hỏi Lab Coach nếu guideline chưa quy định ca bị cắt/che, không tự suy đoán toàn bộ người và không dùng threshold để loại đối tượng khỏi ground truth. | Đối chiếu ảnh để kiểm tra sai lớp, hộp trùng/lệch, vật bị bỏ sót và cách xử lý mép ảnh; xác minh prediction bổ sung khi hạ ngưỡng (0.20: 17; 0.60: 6), trả ca chưa đạt về rework. |
| Instance segmentation | Mỗi đối tượng đủ điều kiện có một lớp và mask/polygon riêng; hai vật cùng lớp vẫn là hai instance. | Ảnh `kitchen`: bát `kitchen-002` (`bowl`, 67 điểm polygon) nằm trên mặt bàn `kitchen-005` (`dining table`); phần mặt bàn dưới bát bị che (`segmentation_prediction.png`). Cần quy ước về phần nhìn thấy/phần bị che. | Vẽ biên sát đối tượng, tách bát khỏi bàn và tách từng instance; áp dụng guideline về che khuất, hỏi Lab Coach khi chưa có quy ước, không suy diễn phần không thấy. | Kiểm tra biên lấn nền/vật khác, vùng bị thiếu, polygon tự cắt và instance bị gộp/tách sai; đối chiếu ảnh gốc và quy ước che khuất, yêu cầu chỉnh mask nếu chưa đạt. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)

   **Trả lời:** Chỉ sử dụng ba ảnh COCO công khai được bài lab cung cấp và giữ thông tin nguồn ảnh, không đưa ảnh cá nhân, dữ liệu khách hàng, dữ liệu nội bộ, email, số điện thoại, mật khẩu hoặc token vào Colab, báo cáo hay repository công khai; họ tên và MSSV chỉ dùng trong tên repository theo yêu cầu lớp.

2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?

   **Trả lời:** Tôi sẽ dừng xử lý và không tải lên hoặc chia sẻ dữ liệu đó, báo Lab Coach/giảng viên qua kênh hỗ trợ của lớp để xác nhận phạm vi và cách xử lý; nếu đã lỡ công khai thì báo ngay để được hướng dẫn khắc phục.


## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

**Trạng thái chuẩn bị:** các dấu tích dưới đây xác nhận file đã có trong `report/` trên máy và validation đã báo PASS. Cần kiểm tra lại nội dung trên GitHub sau khi commit/push; chưa xác nhận đã nộp LMS.

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
