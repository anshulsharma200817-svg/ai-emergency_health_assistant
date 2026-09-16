import { useState } from "react";
import "./App.css";

interface RegisteredUser {
  id: number;
  email: string;
  first_name: string;
  last_name: string;
  date_joined: string;
}

interface StatusState {
  type: "success" | "error";
  message: string;
  user?: RegisteredUser;
}

function App() {
  const [formData, setFormData] = useState({
    email: "",
    password: "",
    first_name: "",
    last_name: "",
  });

  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState<StatusState | null>(null);

  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setFormData({
      ...formData,
      [event.target.name]: event.target.value,
    });
  };

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setLoading(true);
    setStatus(null);

    try {
      // Endpoint /api/auth/register/ is proxied by Vite to http://127.0.0.1:8080
      const apiUrl = "/api/auth/register/";
      const response = await fetch(apiUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      });

      const data = await response.json();

      console.log("Status Code:", response.status);
      console.log("Response Data:", data);

      if (response.ok) {
        setStatus({
          type: "success",
          message: "Registration successful! Backend logged user details in terminal.",
          user: data,
        });
        // Reset form
        setFormData({
          email: "",
          password: "",
          first_name: "",
          last_name: "",
        });
      } else {
        let errText = "Registration failed.";
        if (typeof data === "object" && data !== null) {
          const keys = Object.keys(data);
          if (keys.length > 0) {
            const firstKey = keys[0];
            const val = data[firstKey];
            errText = `${firstKey}: ${Array.isArray(val) ? val.join(" ") : val}`;
          }
        }
        setStatus({
          type: "error",
          message: errText,
        });
      }
    } catch (error) {
      console.error("Request failed:", error);
      setStatus({
        type: "error",
        message: "Failed to connect to backend server. Make sure Django server is running on port 8080.",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-container">
      <div className="auth-card">
        <div className="auth-header">
          <div className="auth-badge">Fullstack Auth System</div>
          <h1>User Registration</h1>
          <p className="auth-subtitle">
            Submissions are sent to Django API & printed live in the server terminal.
          </p>
        </div>

        {status && (
          <div className={`status-banner ${status.type}`}>
            <div className="status-header">
              <span>{status.type === "success" ? "✅ Success" : "⚠️ Error"}</span>
            </div>
            <div className="status-text">{status.message}</div>
            {status.user && (
              <div className="registered-details">
                <div><strong>User ID:</strong> {status.user.id}</div>
                <div><strong>Email:</strong> {status.user.email}</div>
                <div><strong>Name:</strong> {status.user.first_name || status.user.last_name ? `${status.user.first_name} ${status.user.last_name}` : "Not provided"}</div>
              </div>
            )}
          </div>
        )}

        <form onSubmit={handleSubmit} className="auth-form">
          <div className="form-row">
            <div className="form-group">
              <label htmlFor="first_name">First Name</label>
              <input
                id="first_name"
                name="first_name"
                type="text"
                placeholder="John"
                value={formData.first_name}
                onChange={handleChange}
                disabled={loading}
              />
            </div>

            <div className="form-group">
              <label htmlFor="last_name">Last Name</label>
              <input
                id="last_name"
                name="last_name"
                type="text"
                placeholder="Doe"
                value={formData.last_name}
                onChange={handleChange}
                disabled={loading}
              />
            </div>
          </div>

          <div className="form-group">
            <label htmlFor="email">
              Email Address <span className="required">*</span>
            </label>
            <input
              id="email"
              type="email"
              name="email"
              placeholder="user@example.com"
              value={formData.email}
              onChange={handleChange}
              required
              disabled={loading}
            />
          </div>

          <div className="form-group">
            <label htmlFor="password">
              Password <span className="required">*</span>
            </label>
            <input
              id="password"
              type="password"
              name="password"
              placeholder="••••••••"
              value={formData.password}
              onChange={handleChange}
              required
              minLength={6}
              disabled={loading}
            />
          </div>

          <button type="submit" className="submit-btn" disabled={loading}>
            {loading ? "Registering..." : "Register"}
          </button>
        </form>
      </div>
    </div>
  );
}

export default App;