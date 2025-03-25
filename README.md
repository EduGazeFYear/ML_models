# Face Recognition Attendance System

## Overview
This project is a **real-time face recognition attendance system** that uses a webcam to detect, recognize, and log students' attendance. The system identifies known faces and logs their attendance in an Excel file while also handling unknown faces by saving their images for further review.

## Features
- **Real-Time Face Recognition**: Uses `face_recognition` and `OpenCV` to detect faces via webcam.
- **Attendance Logging**: Records recognized faces into an `attendance.xlsx` file.
- **Unknown Face Handling**: Saves unknown faces in a dedicated folder (`unknown_faces/`) to prevent repeated entries.
- **Efficient Recognition**: Ensures that each student is logged only once per day.
- **Matplotlib Display**: Shows real-time frames with bounding boxes and labels.

## System Flow
1. **Load Encoded Faces**: The system loads pre-stored face encodings from `encodings.json`.
2. **Start Webcam Feed**: Captures frames continuously using OpenCV.
3. **Detect Faces**: Identifies faces in the current frame and extracts encodings.
4. **Match Faces**:
   - If a match is found, attendance is logged (if not already recorded that day).
   - If no match is found, the face is saved as an unknown entry.
5. **Update Attendance Log**: Saves attendance details in an Excel file.
6. **Display Video Feed**: Frames are updated dynamically using Matplotlib.
7. **Exit System**: The system stops when the user presses a key.

## Dependencies
Ensure you have the following Python libraries installed:
```sh
pip install opencv-python face-recognition numpy pandas matplotlib openpyxl
```

## Folder Structure
```
📂 Project Folder
├── encodings.json             # Stored face encodings
├── attendance.xlsx            # Attendance record
├── unknown_faces/             # Folder for unknown face images
├── main.py                    # The main script
```

## How to Run
1. **Train the System**:
   - Store known face encodings using the dataset processing script.
   - Run `encodings.json` generation to store recognized face data.

2. **Start the Attendance System**:
   - Run the script:
   ```sh
   python main.py
   ```
   - The webcam will open, and the system will recognize faces in real time.
   - Press any key to exit.

## Possible Improvements
- **Improve Face Matching Accuracy**: Use deep learning models like FaceNet.
- **Optimize Performance**: Reduce frame processing rate for efficiency.
- **Implement GUI**: Develop a frontend interface for better usability.
- **Mobile App Integration**: Sync attendance with a mobile app.

## Conclusion
This face recognition attendance system is **efficient and scalable** for classrooms and similar environments. By leveraging `face_recognition` and `OpenCV`, it ensures seamless attendance tracking while handling unknown faces. Future improvements could focus on deep learning-based models for enhanced accuracy.

