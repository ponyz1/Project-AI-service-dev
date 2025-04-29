"use client";

import React from 'react';
import axios from 'axios';

export default function App() {

  const test = async () => {
    try {
      const response = await axios.post('http://localhost:8000/test');
      console.log('Backend response:', response.data);
      alert(`Backend response: ${JSON.stringify(response.data)}`);
    } catch (err) {
      console.error('Error connecting to backend:', err);
      alert('Failed to connect to backend');
    }
  };
  const testget = async () => {
    try {
      const response = await axios.get('http://localhost:8000/');
      console.log('Backend response:', response.data);
      alert(`Backend response: ${JSON.stringify(response.data)}`);
    } catch (err) {
      console.error('Error connecting to backend:', err);
      alert('Failed to connect to backend');
    }
  };
  

  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-4 bg-gray-50">
      <h1 className="text-2xl font-bold mb-6">🍽️ Food Matching AI</h1>
      <button
        onClick={testget}
        className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
      >
      </button>
    </div>
  );
}
