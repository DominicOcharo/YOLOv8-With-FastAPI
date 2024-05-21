async function uploadImage() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];

    if (!file) {
        alert('Please select an image.');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('http://127.0.0.1:8080/yolo', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Error analyzing image.');
        }

        const data = await response.json();
        const imageId = data.id;
        const labels = data.labels;
        displayResult(file, imageId, labels);
    } catch (error) {
        alert(error.message);
    }
}

function displayResult(uploadedImage, imageId, labels) {
    const resultDiv = document.getElementById('result');
    resultDiv.innerHTML = '';  // Clear previous results

    // Display uploaded image
    const uploadedImg = document.createElement('img');
    uploadedImg.src = URL.createObjectURL(uploadedImage);
    uploadedImg.alt = "Uploaded Image";
    resultDiv.appendChild(uploadedImg);

    // Display detected image
    const detectedImg = document.createElement('img');
    detectedImg.src = `http://127.0.0.1:8080/yolo/${imageId}`;
    detectedImg.alt = "Detected Image";
    resultDiv.appendChild(detectedImg);

    // Display detected labels
    const labelsParagraph = document.createElement('p');
    labelsParagraph.textContent = `Detected Labels: ${labels.join(', ')}`;
    resultDiv.appendChild(labelsParagraph);
}
