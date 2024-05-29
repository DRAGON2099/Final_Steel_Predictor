document
  .getElementById("searchForm")
  .addEventListener("submit", function (event) {
    event.preventDefault(); // Prevent form submission

    // Get user input values
    const yield_strength = document.getElementById("yield_strength").value;
    const uts = document.getElementById("uts").value;

    // Construct JSON data to send to server
    const inputData = {
      yield_strength: parseFloat(yield_strength),
      uts: parseFloat(uts),
      // Add more properties as needed
    };

    // Send input data to server for prediction
    fetch(" http://127.0.0.1:5000/steel_treat", {
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
          <p>Treatment Prediction: ${data.heat_treat_Prediction}</p>
      `;
      })
      .catch((error) => {
        console.error("Error:", error);
      });
  });
