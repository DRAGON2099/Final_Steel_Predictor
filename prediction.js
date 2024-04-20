document
  .getElementById("predictionForm")
  .addEventListener("submit", function (event) {
    event.preventDefault(); // Prevent form submission

    // Get user input values
    const carbonPercent = document.getElementById("carbonPercent").value;
    const chromiumPercent = document.getElementById("chromiumPercent").value;
    const aluminiumPercent = document.getElementById("aluminiumPercent").value;
    const manganesePercent = document.getElementById("manganesePercent").value;
    const siliconPercent = document.getElementById("siliconPercent").value;
    const nickelPercent = document.getElementById("nickelPercent").value;
    const cobaltPercent = document.getElementById("cobaltPercent").value;
    const molybdenumPercent =
      document.getElementById("molybdenumPercent").value;
    const tungstenPercent = document.getElementById("tungstenPercent").value;
    const niobiumPercent = document.getElementById("niobiumPercent").value;
    const phosphorousPercent =
      document.getElementById("phosphorousPercent").value;
    const copperPercent = document.getElementById("copperPercent").value;
    const titaniumPercent = document.getElementById("titaniumPercent").value;
    const tantalumPercent = document.getElementById("tantalumPercent").value;
    const hafniumPercent = document.getElementById("hafniumPercent").value;
    const rheniumPercent = document.getElementById("rheniumPercent").value;
    const vanadiumPercent = document.getElementById("vanadiumPercent").value;
    const boronPercent = document.getElementById("boronPercent").value;
    const nitrogenPercent = document.getElementById("nitrogenPercent").value;
    const oxygenPercent = document.getElementById("oxygenPercent").value;
    const sulphurPercent = document.getElementById("sulphurPercent").value;

    // Construct JSON data to send to server
    const inputData = {
      carbonPercent: parseFloat(carbonPercent),
      chromiumPercent: parseFloat(chromiumPercent),
      aluminiumPercent: parseFloat(aluminiumPercent),
      manganesePercent: parseFloat(manganesePercent),
      siliconPercent: parseFloat(siliconPercent),
      nickelPercent: parseFloat(nickelPercent),
      cobaltPercent: parseFloat(cobaltPercent),
      molybdenumPercent: parseFloat(molybdenumPercent),
      tungstenPercent: parseFloat(tungstenPercent),
      niobiumPercent: parseFloat(niobiumPercent),
      phosphorousPercent: parseFloat(phosphorousPercent),
      copperPercent: parseFloat(copperPercent),
      titaniumPercent: parseFloat(titaniumPercent),
      tantalumPercent: parseFloat(tantalumPercent),
      hafniumPercent: parseFloat(hafniumPercent),
      rheniumPercent: parseFloat(rheniumPercent),
      vanadiumPercent: parseFloat(vanadiumPercent),
      boronPercent: parseFloat(boronPercent),
      nitrogenPercent: parseFloat(nitrogenPercent),
      oxygenPercent: parseFloat(oxygenPercent),
      sulphurPercent: parseFloat(sulphurPercent),
      // Add more properties as needed
    };

    // Send input data to server for prediction
    fetch(" http://127.0.0.1:5000/predict", {
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
          <p>Yield Strength Prediction: ${data.yieldStrengthPrediction}</p>
          <p>UTS Prediction: ${data.utsPrediction}</p>
          <p>Elongation Prediction: ${data.elongationPrediction}</p>
          <p>Reduction in Area Prediction: ${data.reductionAreaPrediction}</p>
          <p>Accuracy (R2 Score): ${data.accuracy}</p>
      `;
      })
      .catch((error) => {
        console.error("Error:", error);
      });
  });
