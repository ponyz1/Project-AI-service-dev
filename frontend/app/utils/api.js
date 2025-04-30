export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || '/api';

export async function getFoods(searchTerm = '') {
  const response = await fetch(`${API_BASE_URL}/foods?search=${encodeURIComponent(searchTerm)}`);
  if (!response.ok) {
    throw new Error('Failed to fetch foods');
  }
  return await response.json();
}

export async function getFoodTypes() {
  const response = await fetch(`${API_BASE_URL}/food-types`);
  if (!response.ok) {
    throw new Error('Failed to fetch food types');
  }
  return await response.json();
}

export async function matchFoods(food, foodType) {
  const response = await fetch(`${API_BASE_URL}/match`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ food, foodType }),
  });
  
  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || 'Failed to match foods');
  }
  
  return await response.json();
}

export default {
    getFoods,
    getFoodTypes,
    matchFoods
};