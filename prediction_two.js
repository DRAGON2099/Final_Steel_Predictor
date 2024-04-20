document
  .getElementById("searchForm")
  .addEventListener("submit", function (event) {
    event.preventDefault(); // Prevent form submission

    // Get user input values
    const yield_strength = document.getElementById("yield_strength").value;
    const uts = document.getElementById("uts").value;
    const elongation = document.getElementById("elongation").value;
    const reduction_area = document.getElementById("reduction_area").value;

    // Construct JSON data to send to server
    const inputData = {
      yield_strength: parseFloat(yield_strength),
      uts: parseFloat(uts),
      elongation: parseFloat(elongation),
      reduction_area: parseFloat(reduction_area),
      // Add more properties as needed
    };

    // Send input data to server for prediction
    fetch(" http://127.0.0.1:5000/steel_search", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(inputData),
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error("Network response was not ok");
        }
        return response.json();
      })
      .then((data) => {
        // Display prediction results on the web page
        const resultsDiv = document.getElementById("predictionResults");
        resultsDiv.innerHTML = `
          <h2>Prediction Results</h2>
          <p>Carbon Prediction: ${data.carbon_percent_Prediction}</p>
          <p>Chromium Prediction: ${data.chromium_percent_Prediction}</p>
          <p>Manganese Prediction: ${data.manganese_percent_Prediction}</p>
          <p>Silicon Prediction: ${data.silicon_percent_Prediction}</p>
          <p>Nickel Prediction: ${data.nickel_percent_Prediction}</p>
          <p>Cobalt Prediction: ${data.cobalt_percent_Prediction}</p>
          <p>Molybdenum Prediction: ${data.molybdenum_percent_Prediction}</p>
          <p>Tungsten Prediction: ${data.tungsten_percent_Prediction}</p>
          <p>Niobium Prediction: ${data.niobium_percent_Prediction}</p>
          <p>Aluminium Prediction: ${data.aluminium_percent_Prediction}</p>
          <p>Phosphorous Prediction: ${data.phosphorous_percent_Prediction}</p>
          <p>Copper Prediction: ${data.copper_percent_Prediction}</p>
          <p>Titanium Prediction: ${data.titanium_percent_Prediction}</p>
          <p>Tantalum Prediction: ${data.tantalum_percent_Prediction}</p>
          <p>Hafnium Prediction: ${data.hafnium_percent_Prediction}</p>
          <p>Rhenium Prediction: ${data.rhenium_percent_Prediction}</p>
          <p>Vanadium Prediction: ${data.vanadium_percent_Prediction}</p>
          <p>Boron Prediction: ${data.boron_percent_Prediction}</p>
          <p>Nitrogen Prediction: ${data.nitrogen_percent_Prediction}</p>
          <p>Oxygen Prediction: ${data.oxygen_percent_Prediction}</p>
          <p>Sulphur Prediction: ${data.sulphur_percent_Prediction}</p>
          <p>Accuracy (R2 Score) Carbon: ${data.carbon_percentScore}</p>
          <p>Accuracy (R2 Score) Chromium: ${data.chromium_percentScore}</p>
          <p>Accuracy (R2 Score) Aluminium: ${data.aluminium_percentScore}</p>
          <p>Accuracy (R2 Score) Manganese: ${data.manganese_percentScore}</p>
          <p>Accuracy (R2 Score) Nickel: ${data.nickel_percentScore}</p>
          <p>Accuracy (R2 Score) Copper: ${data.copper_percentScore}</p>
          <p>Accuracy (R2 Score) Silicon: ${data.silicon_percentScore}</p>
          <p>Accuracy (R2 Score) Titanium: ${data.titanium_percentScore}</p>
          <p>Accuracy (R2 Score) Phosphorous: ${data.phosphorous_percentScore}</p>
          <p>Accuracy (R2 Score) Sulphur: ${data.sulphur_percentScore}</p>
          <p>Accuracy (R2 Score) Nitrogen: ${data.nitrogen_percentScore}</p>
          <p>Accuracy (R2 Score) Vanadium: ${data.vanadium_percentScore}</p>
          <p>Accuracy (R2 Score) Boron: ${data.boron_percentScore}</p>
          <p>Accuracy (R2 Score) Oxygen: ${data.oxygen_percentScore}</p>
          <p>Accuracy (R2 Score) Rhenium: ${data.rhenium_percentScore}</p>
          <p>Accuracy (R2 Score) Hafnium: ${data.hafnium_percentScore}</p>
          <p>Accuracy (R2 Score) Tantalum: ${data.tantalum_percentScore}</p>
          <p>Accuracy (R2 Score) Cobalt: ${data.cobalt_percentScore}</p>
          <p>Accuracy (R2 Score) Molybdenum: ${data.molybdenum_percentScore}</p>
          <p>Accuracy (R2 Score) Tungsten: ${data.tungsten_percentScore}</p>
          <p>Accuracy (R2 Score) Niobium: ${data.niobium_percentScore}</p>
      `;
      })
      .catch((error) => {
        console.error("Error:", error);
      });
  });
