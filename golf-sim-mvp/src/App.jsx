
import React, { useState } from "react";
import axios from "axios";
import { Line } from "react-chartjs-2";
import "chart.js/auto";

function App() {
  const [video, setVideo] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const handleUpload = async () => {
    if (!video) {
      setError("Please select a video first!");
      return;
    }
    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append("file", video);
    try {
      const res = await axios.post("http://127.0.0.1:5000/process-video", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResult(res.data);
    } catch (err) {
      setError("Backend not available, showing mock results.");
      const mockResult = {
        speed: 42.5,
        angle: 15.0,
        carry_distance: 160.0,
        apex_height: 18.2,
        hang_time: 4.2,
        trajectory: [
          [0, 0],
          [10, 2],
          [20, 3],
          [30, 4],
          [40, 3],
          [50, 1],
          [60, 0],
        ],
      };
      setResult(mockResult);
    }
    setLoading(false);
  };

  const trajectoryData =
    result && result.trajectory
      ? {
          labels: result.trajectory.map((p) => p[0]),
          datasets: [
            {
              label: "Ball Flight (m vs. height)",
              data: result.trajectory.map((p) => p[1]),
              borderColor: "#2ecc71",
              tension: 0.3,
              fill: false,
            },
          ],
        }
      : null;

  return (
    <div className="container">
      <h1>Golf Swing Analyzer</h1>
      <div className="upload-area">
        <label htmlFor="video-upload">Upload your golf swing video:</label><br />
        <input
          id="video-upload"
          type="file"
          accept="video/*"
          onChange={(e) => setVideo(e.target.files[0])}
        />
        <br />
        <button
          onClick={handleUpload}
          disabled={!video || loading}
          style={{
            marginTop: "16px",
            padding: "8px 24px",
            background: video ? "#388e3c" : "#888",
            color: "white",
            border: "none",
            borderRadius: "5px",
            cursor: video ? "pointer" : "not-allowed",
            fontWeight: 500,
          }}
        >
          {loading ? "Processing..." : "Submit & Analyze"}
        </button>
      </div>
      <div className="metrics" id="metrics-area">
        <h2>Metrics</h2>
        <p>Ball Speed: <span id="ball-speed">{result ? result.speed.toFixed(2) : "--"}</span> mph</p>
        <p>Launch Angle: <span id="launch-angle">{result ? result.angle.toFixed(2) : "--"}</span>°</p>
        <p>Carry Distance: <span id="carry-distance">{result ? result.carry_distance.toFixed(2) : "--"}</span> yards</p>
        <p>Apex Height: <span id="apex-height">{result ? result.apex_height?.toFixed(2) : "--"}</span> ft</p>
        <p>Hang Time: <span id="hang-time">{result ? result.hang_time?.toFixed(2) : "--"}</span> s</p>
        {trajectoryData && (
          <div style={{ width: "100%", margin: "20px auto" }}>
            <Line data={trajectoryData} />
          </div>
        )}
        {result && result.processed_video_url && (
          <div style={{ marginTop: "24px" }}>
            <h3>Processed Video</h3>
            <video controls width="100%">
              <source src={`http://127.0.0.1:5000${result.processed_video_url}`} type="video/mp4" />
              Your browser does not support the video tag.
            </video>
            <br />
            <a href={`http://127.0.0.1:5000${result.processed_video_url}`} download>
              Download Processed Video
            </a>
          </div>
        )}
      </div>
      {error && <p style={{ color: "red", marginTop: "16px" }}>{error}</p>}
    </div>
  );
}

export default App;
