"use client";

import React, { useState, useEffect } from 'react';
import { getFoods, getFoodTypes, matchFoods } from './utils/api';
import axios from 'axios';

export default function App() {
  const [searchTerm, setSearchTerm] = useState('');
  const [foods, setFoods] = useState([]);
  const [selectedFood, setSelectedFood] = useState(null);
  const [foodTypes, setFoodTypes] = useState([]);
  const [selectedType, setSelectedType] = useState('');
  const [matches, setMatches] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [showDropdown, setShowDropdown] = useState(false);


  useEffect(() => {
    const loadFoodTypes = async () => {
      try {
        const types = await getFoodTypes();
        setFoodTypes(types);
        if (types.length > 0) {
          setSelectedType(types[0]);
        }
      } catch (err) {
        console.error(err);
      }
    };
    
    loadFoodTypes();
  }, []);


  useEffect(() => {
    const loadFoods = async () => {
      try {
        if (searchTerm.length === 0) {
          setFoods([]);
          setShowDropdown(false);
          return;
        }
        const foodData = await getFoods(searchTerm);
        setFoods(foodData);
        setShowDropdown(true);
      } catch (err) {
        console.error(err);
      }
    };
    loadFoods();
  }, [searchTerm]);

  const handleSearchChange = (e) => {
    setSearchTerm(e.target.value);
    setSelectedFood(null);
  };

  const handleFoodSelect = (food) => {
    setShowDropdown(false);
    setSelectedFood(food);
    setSearchTerm("-"+food.name+"-");
  };

  const handleTypeChange = (e) => {
    setSelectedType(e.target.value);
  };

  const findMatches = async () => {
    if (!selectedFood || !selectedType) {
      alert('Please select both a food and a type to match with');
      return;
    }

    setIsLoading(true);
    try {
      const result = await matchFoods(selectedFood.name, selectedType);
      console.log('Matches found:', result.matched_foods);
      setMatches(result.matched_foods);
    } catch (error) {
      console.error('Error finding matches:', error);
      alert('Failed to find matches');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-4 bg-gray-50">
      <div className="w-full max-w-md bg-white rounded-lg shadow-md p-6">
        <h1 className="text-2xl font-bold mb-6 text-center text-blue-600">🍽️ Food Matching AI</h1>
        
        {/* Food Search */}
        <div className="mb-4 relative">
          <label className="block text-sm font-medium text-black mb-1">Search for a food:</label>
          <input
            type="text"
            value={searchTerm}
            onChange={handleSearchChange}
            placeholder="Type to search foods..."
            className="text-black w-full px-4 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          {showDropdown && foods.length > 0 && (
            <ul className="absolute z-10 w-full mt-1 bg-white border border-gray-300 rounded-md shadow-lg max-h-60 overflow-auto">
              {foods.map((food, index) => (
                <li
                  key={index}
                  className="px-4 py-2 hover:bg-blue-50 cursor-pointer"
                  onClick={() => handleFoodSelect(food)}
                >
                  <div className="font-medium text-black">{food.name}</div>
                  <div className="text-sm text-black">{food.type}</div>
                </li>
              ))}
            </ul>
          )}
        </div>

        {/* Selected Food Display */}
        {selectedFood && (
          <div className="mb-4 p-3 bg-blue-50 rounded-md">
            <h3 className="text-black font-medium">Selected Food:</h3>
            <p  className="text-black">{selectedFood.name} <span className="text-sm text-black">({selectedFood.type})</span></p>
          </div>
        )}

        {/* Type Selection */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-black mb-1">Match with food type:</label>
          <select
            value={selectedType}
            onChange={handleTypeChange}
            className="text-gray-600 w-full px-4 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option className="text-black" value="">Select a type</option>
            {foodTypes.map((type, index) => (
              <option className="text-black" key={index} value={type}>{type}</option>
            ))}
          </select>
        </div>

        {/* Find Matches Button */}
        <button
          onClick={findMatches}
          disabled={!selectedFood || !selectedType || isLoading}
          className={`w-full py-2 px-4 rounded-md text-white font-medium ${(!selectedFood || !selectedType || isLoading) ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700'}`}
        >
          {isLoading ? 'Finding Matches...' : 'Find Matching Foods'}
        </button>

        {/* Results */}
        {matches.length > 0 && (
          <div className="mt-6">
            <h3 className="text-black text-lg font-medium mb-3">Recommended Matches:</h3>
            <ul className="space-y-3">
              {matches.map((match, index) => (
                <li key={index} className="p-3 bg-gray-50 rounded-md border border-gray-200">
                  <div className="flex justify-between items-center">
                    <span className="text-black font-medium">{match.name}</span>
                    <span className="text-sm bg-blue-100 text-blue-800 px-2 py-1 rounded">
                      Score: {match.score.toFixed(2)}
                    </span>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}