import * as tf from '@tensorflow/tfjs';

async function singlePrediction(imageElement) {
    try {

        const model = await tf.loadLayersModel('./20_Nov_2nd_TFJS/model.json'); 

        let tensor = tf.browser.fromPixels(imageElement)
            .resizeNearestNeighbor([224, 224])
            .toFloat()
            .div(tf.scalar(255))
            .expandDims(); 

        const predictions = await model.predict(tensor).data();

        const highestProbabilityIndex = predictions.indexOf(Math.max(...predictions));
        const highestProbability = predictions[highestProbabilityIndex];

        if (highestProbability >= 0.65) {
            const classId = highestProbabilityIndex + 1;
            return { predictedClass: classId, probabilities: predictions };
        } else {
            return { predictedClass: 0, probabilities: predictions }; 
        }
    } catch (error) {
        console.error("Error during prediction:", error);
        return null;
    }
}

export { singlePrediction };
