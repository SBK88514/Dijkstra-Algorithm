# Solving Mazes Using Dijkstra's Algorithm

## 📌 Project Description
This project focuses on finding the shortest path in a maze using **Dijkstra's algorithm**. The maze is represented as an image in the form of a pixel matrix, where each pixel signifies part of the maze:

- 🟩 **Black pixels** represent walls and cannot be traversed.
- 🔵 **White pixels** represent open paths.

The task is to implement **Dijkstra's algorithm in Python** and use it to determine the shortest path from the maze's start point to its end point.

---

## 🛠️ Implementation Steps
1. **Loading the Maze Image** – Open and display the maze image using OpenCV.
2. **Identifying Start and End Points** – Analyze the image to determine the maze's entry and exit points.
3. **Converting the Maze to a Matrix** – Represent the maze as a numerical matrix where walls are marked as impassable.
4. **Implementing Dijkstra's Algorithm** – Compute the shortest path from the start to the end point.
5. **Displaying the Solution Path** – Mark the computed path directly on the maze image and visualize it.

---

## 🖥️ Technologies & Tools Used
- 🐝 **Python** – The programming language used for implementation.
- 🎨 **OpenCV** – For loading and displaying the maze image.
- 📌 **Heapq** – To efficiently manage the priority queue in Dijkstra's algorithm.
- 🔢 **NumPy** – For handling the maze matrix.

---

## 🎯 Expected Output
The project successfully finds the **shortest path** in the maze and **visually displays the computed path** on the image. The highlighted path clearly indicates the **route from the starting point to the exit**.

---

## 🔧 Installation & Running Locally
To download and run this project on your local machine, follow these steps:

### **1️⃣ Install Python**  
Ensure you have Python (version **3.8 or higher**) installed. You can download it from the official site:  
[Python Official Download](https://www.python.org/downloads/)

### **2️⃣ Clone the Repository**  
Download the project files via GitHub or clone the repository using:
```bash
git clone https://github.com/SBK88514/Dijkstra-Algorithm.git
cd Dijkstra-Algorithm
```

### **3️⃣ Install Dependencies**  
Run the following command to install all required dependencies:
```bash
pip install -r requirements.txt
```

### **4️⃣ Run the Solution Script**  
Execute the main script with:
```bash
python src/maze_solver.py --input mazes/maze1.png --output results/solved_maze1.png
```

📌 **After execution**, the maze image with the highlighted shortest path **will be saved** in the `results` directory.

---

## 📝 License
This project is licensed under the **MIT License** – feel free to use and modify it.

---

## ✨ Contributing
Contributions are welcome! If you'd like to improve this project:
1. **Fork the repository**  
2. **Create a new branch** (`feature-name`)  
3. **Make your changes**  
4. **Submit a pull request** 🎉

---

## 🔗 Contact
For any questions or collaboration opportunities, feel free to reach out.

---
