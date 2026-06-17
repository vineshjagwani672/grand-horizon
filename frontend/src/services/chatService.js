import axios from "axios";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function sendMessage(question) {
  const response = await axios.post(`${API_URL}/chat`, { question });
  return response.data;
}
