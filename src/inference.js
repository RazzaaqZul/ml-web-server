// untuk load dan prediksi model.
const tfjs = require("@tensorflow/tfjs-node");

// Load model tensorflow.js
function loadModel() {
  const modelUrl = "file://models/model.json";
  /*
 memuat (load) suatu model neural network. 
 Method ini dapat digunakan untuk load model yang dibangun berdasarkan tf.layers.*, tf.sequential(), tf.model(), 
 dan model yang disimpan menggunakan method tf.LayersModel.save().
 */
  return tfjs.loadLayersModel(modelUrl);
}

// predict() untuk memprediksi data (berupa imageBuffer) dengan model yang telah di-load.
function predict(model, imageBuffer) {
  const tensor = tfjs.node
    //  Decoding atau decode : proses mengubah pesan kode menjadi bahasa yang mudah dimengerti oleh mesin.
    //  Buffer : objek yang digunakan untuk merepresentasikan sejumlah data biner atau byte.
    .decodeJpeg(imageBuffer)
    //  mengubah gambar yang sebelumnya di-decode
    .resizeNearestNeighbor([150, 150])
    //  menambahkan dimensi ekstra pada tensor serta mengonversi nilai-nilai dalam tensor tersebut ke tipe data float.
    .expandDims()
    .toFloat();

  //  mengembalikan hasil prediksi dari tensor yang telah di-preprocessing.
  return model.predict(tensor).data();
}

module.exports = { loadModel, predict };
