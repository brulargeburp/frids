/*
 * Copyright 2023 Shubham Panchal
 * Licensed under the Apache License, Version 2.0 (the "License");
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.ml.quaterion.facenetdetection

import android.Manifest
import android.util.Log
import android.widget.Toast
import com.android.volley.toolbox.StringRequest
import com.android.volley.toolbox.Volley
import android.content.Context
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.media.ExifInterface
import android.widget.EditText
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.DocumentsContract
import android.text.method.ScrollingMovementMethod
import android.util.Size
import android.view.View
import android.view.WindowInsets
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.documentfile.provider.DocumentFile
import androidx.lifecycle.LifecycleOwner
import com.google.common.util.concurrent.ListenableFuture
import com.ml.quaterion.facenetdetection.databinding.ActivityMainBinding
import com.ml.quaterion.facenetdetection.model.FaceNetModel
import com.ml.quaterion.facenetdetection.model.Models
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import java.io.*
import java.util.concurrent.Executors
import java.net.URLEncoder
import android.graphics.BitmapFactory
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity() {

    private var currentMode = "none" // "in", "out", or "none"
    private var isSerializedDataStored = false

    // Serialized data will be stored ( in app's private storage ) with this filename.
    private val SERIALIZED_DATA_FILENAME = "image_data"

    // Shared Pref key to check if the data was stored.
    private val SHARED_PREF_IS_DATA_STORED_KEY = "is_data_stored"

    private lateinit var activityMainBinding : ActivityMainBinding
    private lateinit var previewView : PreviewView
    private lateinit var frameAnalyser  : FrameAnalyser
    private lateinit var faceNetModel : FaceNetModel
    private lateinit var fileReader : FileReader
    private lateinit var cameraProviderFuture : ListenableFuture<ProcessCameraProvider>
    private lateinit var sharedPreferences: SharedPreferences
    private lateinit var detector: com.google.mlkit.vision.face.FaceDetector

    // <----------------------- User controls --------------------------->

    // Use the device's GPU to perform faster computations.
    // Refer https://www.tensorflow.org/lite/performance/gpu
    private val useGpu = true

    // Use XNNPack to accelerate inference.
    // Refer https://blog.tensorflow.org/2020/07/accelerating-tensorflow-lite-xnnpack-integration.html
    private val useXNNPack = true

    // You may the change the models here.
    // Use the model configs in Models.kt
    // Default is Models.FACENET ; Quantized models are faster
    // 2/2/2026: Changed model to MOBILE_FACENET
    private val modelInfo = Models.MOBILE_FACENET

    // Camera Facing
    private var cameraFacing = CameraSelector.LENS_FACING_BACK
    private val PREF_URL_KEY = "base_url_gsheets"
    private val DEFAULT_URL = "https://script.google.com/macros/s/PASTE_YOUR_ID_HERE/exec"

    // <---------------------------------------------------------------->


    companion object {

        lateinit var logTextView : TextView

        fun setMessage( message : String ) {
            // This forces the text update to happen on the correct thread
            (logTextView.context as MainActivity).runOnUiThread {
                logTextView.text = message
            }
        }

    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Remove the status bar to have a full screen experience
        // See this answer on SO -> https://stackoverflow.com/a/68152688/10878733
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            window.decorView.windowInsetsController!!
                .hide( WindowInsets.Type.statusBars() or WindowInsets.Type.navigationBars())
        }
        else {
            window.decorView.systemUiVisibility = View.SYSTEM_UI_FLAG_FULLSCREEN
        }
        activityMainBinding = ActivityMainBinding.inflate( layoutInflater )
        setContentView( activityMainBinding.root )

        previewView = activityMainBinding.previewView
        logTextView = activityMainBinding.logTextview
        logTextView.movementMethod = ScrollingMovementMethod()
        // Necessary to keep the Overlay above the PreviewView so that the boxes are visible.
        val boundingBoxOverlay = activityMainBinding.bboxOverlay
        boundingBoxOverlay.cameraFacing = cameraFacing
        boundingBoxOverlay.setWillNotDraw( false )
        boundingBoxOverlay.setZOrderOnTop( true )

        faceNetModel = FaceNetModel( this , modelInfo , useGpu , useXNNPack )
        frameAnalyser = FrameAnalyser( this , boundingBoxOverlay , faceNetModel )
        fileReader = FileReader( faceNetModel )

        val options = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .build()
        detector = FaceDetection.getClient(options)

        // Flip camera button listener
        activityMainBinding.flipCameraButton.setOnClickListener {
            cameraFacing = if ( cameraFacing == CameraSelector.LENS_FACING_BACK ) {
                CameraSelector.LENS_FACING_FRONT
            } else {
                CameraSelector.LENS_FACING_BACK
            }
            boundingBoxOverlay.cameraFacing = cameraFacing
            val cameraProvider = cameraProviderFuture.get()
            cameraProvider.unbindAll()
            bindPreview( cameraProvider )
        }

        // We'll only require the CAMERA permission from the user.
        // For scoped storage, particularly for accessing documents, we won't require WRITE_EXTERNAL_STORAGE or
        // READ_EXTERNAL_STORAGE permissions. See https://developer.android.com/training/data-storage
        if ( ActivityCompat.checkSelfPermission( this , Manifest.permission.CAMERA ) != PackageManager.PERMISSION_GRANTED ) {
            requestCameraPermission()
        }
        else {
            startCameraPreview()
        }

        sharedPreferences = getSharedPreferences( getString( R.string.app_name ) , Context.MODE_PRIVATE )
        isSerializedDataStored = sharedPreferences.getBoolean( SHARED_PREF_IS_DATA_STORED_KEY , false )

        val internalImagesDir = File(getExternalFilesDir(null), "images")
        if (!internalImagesDir.exists()) internalImagesDir.mkdirs()

        // --- START OF CORRECTED BLOCK ---
        if (!isSerializedDataStored) {
            val images = ArrayList<Pair<String, Bitmap>>()
            val studentFolders = internalImagesDir.listFiles()?.filter { it.isDirectory }

            studentFolders?.forEach { folder ->
                val name = folder.name
                folder.listFiles()?.forEach { file ->
                    if (file.extension.lowercase() == "png" || file.extension.lowercase() == "jpg") {
                        val bitmap = BitmapFactory.decodeFile(file.absolutePath)
                        if (bitmap != null) {
                            images.add(kotlin.Pair(name, bitmap))
                        }
                    }
                }
            }
            // fileReader.run MUST be outside the loops
            fileReader.run(images, fileReaderCallback)
        }
        else {
            val alertDialog = AlertDialog.Builder(this).apply {
                setTitle("Serialized Data")
                setMessage("Existing image data was found on this device. Would you like to load it?")
                setCancelable(false)
                setNegativeButton("LOAD") { dialog, _ ->
                    dialog.dismiss()
                    frameAnalyser.faceList = loadSerializedImageData()
                    Logger.log("Serialized data loaded.")
                }
                setPositiveButton("RESCAN") { dialog, _ ->
                    dialog.dismiss()
                    launchChooseDirectoryIntent()
                }
                create()
            }
            alertDialog.show()
        }

        // ADD YOUR BUTTON LISTENERS HERE
        activityMainBinding.btnTimeIn.setOnClickListener {
            currentMode = "in"
            Toast.makeText(this, "Mode: TIME IN", Toast.LENGTH_SHORT).show()
        }
        activityMainBinding.btnTimeOut.setOnClickListener {
            currentMode = "out"
            Toast.makeText(this, "Mode: TIME OUT", Toast.LENGTH_SHORT).show()
        }

        // Use activityMainBinding to link the buttons
        activityMainBinding.btnTimeIn.setOnClickListener {
            currentMode = "in"
        }

        activityMainBinding.btnTimeOut.setOnClickListener {
            currentMode = "out"
        }

        activityMainBinding.btnSettings.setOnClickListener {
            // 1. Add the new items to this list
            val options = arrayOf(
                "Register New Student",
                "Manage Enrolled Students",
                "Edit Server URL", // NEW OPTION
                "View/Change Image Directory",
                "QR Fallback (AppSheet)",
                "Cancel"
            )

            val builder = AlertDialog.Builder(this)
            builder.setTitle("D-PASSS AI Menu")
            builder.setItems(options) { dialog, which ->
                when (which) {
                    0 -> showRegistrationDialog()
                    1 -> showStudentListDialog() // Ensure this function exists below
                    2 -> showEditUrlDialog()
                    3 -> showDirectoryInfo()      // NEW FUNCTION
                    4 -> launchQRScannerFallback()
                    5 -> dialog.dismiss()
                }
            }
            builder.show()
        }
    }

    // ---------------------------------------------- //

    // Attach the camera stream to the PreviewView.
    private fun startCameraPreview() {
        cameraProviderFuture = ProcessCameraProvider.getInstance( this )
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            bindPreview(cameraProvider) },
            ContextCompat.getMainExecutor(this) )
    }

    private fun bindPreview(cameraProvider : ProcessCameraProvider) {
        val preview : Preview = Preview.Builder().build()
        val cameraSelector : CameraSelector = CameraSelector.Builder()
            .requireLensFacing( cameraFacing )
            .build()
        preview.setSurfaceProvider( previewView.surfaceProvider )
        val imageFrameAnalysis = ImageAnalysis.Builder()
            .setTargetResolution(Size( 480, 640 ) )
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
        imageFrameAnalysis.setAnalyzer(Executors.newSingleThreadExecutor(), frameAnalyser )
        cameraProvider.bindToLifecycle(this as LifecycleOwner, cameraSelector, preview , imageFrameAnalysis  )
    }

    // We let the system handle the requestCode. This doesn't require onRequestPermissionsResult and
    // hence makes the code cleaner.
    // See the official docs -> https://developer.android.com/training/permissions/requesting#request-permission
    private fun requestCameraPermission() {
        cameraPermissionLauncher.launch( Manifest.permission.CAMERA )
    }

    private val cameraPermissionLauncher = registerForActivityResult( ActivityResultContracts.RequestPermission() ) {
        isGranted ->
        if ( isGranted ) {
            startCameraPreview()
        }
        else {
            val alertDialog = AlertDialog.Builder( this ).apply {
                setTitle( "Camera Permission")
                setMessage( "The app couldn't function without the camera permission." )
                setCancelable( false )
                setPositiveButton( "ALLOW" ) { dialog, which ->
                    dialog.dismiss()
                    requestCameraPermission()
                }
                setNegativeButton( "CLOSE" ) { dialog, which ->
                    dialog.dismiss()
                    finish()
                }
                create()
            }
            alertDialog.show()
        }

    }


    // ---------------------------------------------- //


    // Open File chooser to choose the images directory.
    private fun showSelectDirectoryDialog() {
        val alertDialog = AlertDialog.Builder( this ).apply {
            setTitle( "Select Images Directory")
            setMessage( "As mentioned in the project\'s README file, please select a directory which contains the images." )
            setCancelable( false )
            setPositiveButton( "SELECT") { dialog, which ->
                dialog.dismiss()
                launchChooseDirectoryIntent()
            }
            create()
        }
        alertDialog.show()
    }


    private fun launchChooseDirectoryIntent() {
        val intent = Intent( Intent.ACTION_OPEN_DOCUMENT_TREE )
        // startForActivityResult is deprecated.
        // See this SO thread -> https://stackoverflow.com/questions/62671106/onactivityresult-method-is-deprecated-what-is-the-alternative
        directoryAccessLauncher.launch( intent )
    }


    // Read the contents of the select directory here.
    // The system handles the request code here as well.
    // See this SO question -> https://stackoverflow.com/questions/47941357/how-to-access-files-in-a-directory-given-a-content-uri
    private val directoryAccessLauncher = registerForActivityResult( ActivityResultContracts.StartActivityForResult() ) {
        val dirUri = it.data?.data ?: return@registerForActivityResult
        val childrenUri =
            DocumentsContract.buildChildDocumentsUriUsingTree(
                dirUri,
                DocumentsContract.getTreeDocumentId( dirUri )
            )
        val tree = DocumentFile.fromTreeUri(this, childrenUri)
        val images = ArrayList<Pair<String,Bitmap>>()
        var errorFound = false
        if ( tree!!.listFiles().isNotEmpty()) {
            for ( doc in tree.listFiles() ) {
                if ( doc.isDirectory && !errorFound ) {
                    val name = doc.name!!
                    for ( imageDocFile in doc.listFiles() ) {
                        try {
                            images.add( Pair( name , getFixedBitmap( imageDocFile.uri ) ) )
                        }
                        catch ( e : Exception ) {
                            errorFound = true
                            Logger.log( "Could not parse an image in $name directory. Make sure that the file structure is " +
                                    "as described in the README of the project and then restart the app." )
                            break
                        }
                    }
                    Logger.log( "Found ${doc.listFiles().size} images in $name directory" )
                }
                else {
                    errorFound = true
                    Logger.log( "The selected folder should contain only directories. Make sure that the file structure is " +
                            "as described in the README of the project and then restart the app." )
                }
            }
        }
        else {
            errorFound = true
            Logger.log( "The selected folder doesn't contain any directories. Make sure that the file structure is " +
                    "as described in the README of the project and then restart the app." )
        }
        if ( !errorFound ) {
            fileReader.run( images , fileReaderCallback )
            Logger.log( "Detecting faces in ${images.size} images ..." )
        }
        else {
            val alertDialog = AlertDialog.Builder( this ).apply {
                setTitle( "Error while parsing directory")
                setMessage( "There were some errors while parsing the directory. Please see the log below. Make sure that the file structure is " +
                        "as described in the README of the project and then tap RESELECT" )
                setCancelable( false )
                setPositiveButton( "RESELECT") { dialog, which ->
                    dialog.dismiss()
                    launchChooseDirectoryIntent()
                }
                setNegativeButton( "CANCEL" ){ dialog , which ->
                    dialog.dismiss()
                    finish()
                }
                create()
            }
            alertDialog.show()
        }
    }


    // Get the image as a Bitmap from given Uri and fix the rotation using the Exif interface
    // Source -> https://stackoverflow.com/questions/14066038/why-does-an-image-captured-using-camera-intent-gets-rotated-on-some-devices-on-a
    private fun getFixedBitmap( imageFileUri : Uri ) : Bitmap {
        var imageBitmap = BitmapUtils.getBitmapFromUri( contentResolver , imageFileUri )
        val exifInterface = ExifInterface( contentResolver.openInputStream( imageFileUri )!! )
        imageBitmap =
            when (exifInterface.getAttributeInt( ExifInterface.TAG_ORIENTATION ,
                ExifInterface.ORIENTATION_UNDEFINED )) {
                ExifInterface.ORIENTATION_ROTATE_90 -> BitmapUtils.rotateBitmap( imageBitmap , 90f )
                ExifInterface.ORIENTATION_ROTATE_180 -> BitmapUtils.rotateBitmap( imageBitmap , 180f )
                ExifInterface.ORIENTATION_ROTATE_270 -> BitmapUtils.rotateBitmap( imageBitmap , 270f )
                else -> imageBitmap
            }
        return imageBitmap
    }


    // ---------------------------------------------- //


    private val fileReaderCallback = object : FileReader.ProcessCallback {
        override fun onProcessCompleted(data: ArrayList<Pair<String, FloatArray>>, numImagesWithNoFaces: Int) {
            frameAnalyser.faceList = data
            saveSerializedImageData( data )
            Logger.log( "Images parsed. Found $numImagesWithNoFaces images with no faces." )
        }
    }


    private fun saveSerializedImageData(data : ArrayList<Pair<String,FloatArray>> ) {
        val serializedDataFile = File( filesDir , SERIALIZED_DATA_FILENAME )
        ObjectOutputStream( FileOutputStream( serializedDataFile )  ).apply {
            writeObject( data )
            flush()
            close()
        }
        sharedPreferences.edit().putBoolean( SHARED_PREF_IS_DATA_STORED_KEY , true ).apply()
    }

    // This is the bridge that FrameAnalyser will call
    fun onFaceRecognized(name: String) {
        // 1. Capture the 'in' or 'out' value right now
        val actionToLog = currentMode

        if (actionToLog != "none" && name != "Unknown") {
            // 2. Reset the global mode immediately so it doesn't log twice
            currentMode = "none"

            // 3. Send the CAPTURED mode to Google
            logToGSheets(name, "G12-STEM", actionToLog)

            // 4. Update the UI using the captured mode
            runOnUiThread {
                Logger.log("Attendance Logged: $name as $actionToLog")
                Toast.makeText(this, "Logged $name as $actionToLog", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun loadSerializedImageData() : ArrayList<Pair<String,FloatArray>> {
        val serializedDataFile = File( filesDir , SERIALIZED_DATA_FILENAME )
        val objectInputStream = ObjectInputStream( FileInputStream( serializedDataFile ) )
        val data = objectInputStream.readObject() as ArrayList<Pair<String,FloatArray>>
        objectInputStream.close()
        return data
    }

    private fun logToGSheets(name: String, section: String, action: String) {
        val baseUrl = sharedPreferences.getString(PREF_URL_KEY, DEFAULT_URL) ?: DEFAULT_URL

        // ENCODE the name and section to handle spaces/special characters
        val encodedName = URLEncoder.encode(name, "UTF-8")
        val encodedSection = URLEncoder.encode(section, "UTF-8")

        val url = "$baseUrl?action=$action&name=$encodedName&section=$encodedSection"

        Log.d("D-PASS", "Connecting to: $url")

        val request = StringRequest(com.android.volley.Request.Method.GET, url,
            { response ->
                Toast.makeText(this, "Server: $response", Toast.LENGTH_SHORT).show()
            },
            { error ->
                Log.e("D-PASSS", "Error: ${error.message}")
            }
        )
        Volley.newRequestQueue(this).add(request)
    }

    private fun showRegistrationDialog() {
        val builder = AlertDialog.Builder(this)
        val inflater = layoutInflater
        val dialogLayout = inflater.inflate(R.layout.dialog_register, null)

        val etLast = dialogLayout.findViewById<EditText>(R.id.etLastName)
        val etFirst = dialogLayout.findViewById<EditText>(R.id.etFirstName)
        val etMI = dialogLayout.findViewById<EditText>(R.id.etMI)
        val etSection = dialogLayout.findViewById<EditText>(R.id.etSection)

        builder.setView(dialogLayout)
        builder.setPositiveButton("Enroll student and Burst") { _, _ ->
            val fullName = "${etLast.text.toString().uppercase()}, ${etFirst.text.toString().uppercase()} ${etMI.text.toString().uppercase()} (${etSection.text.toString().uppercase()})"
            // Use the current frame from the camera
            val faceBitmap = previewView.bitmap
            startBurstEnrollment(fullName)
            if (faceBitmap != null) {
                registerNewStudent(faceBitmap, fullName)
            } else {
                Toast.makeText(this, "Camera error: Try again", Toast.LENGTH_SHORT).show()
            }
        }
        builder.setNegativeButton("Cancel", null)
        builder.setMessage("Ensure the student is centered in the camera and looking directly at the phone.")
        builder.show()
    }

    private fun registerNewStudent(fullFrameBitmap: Bitmap, fullName: String) {
        val inputImage = InputImage.fromBitmap(fullFrameBitmap, 0)

        // 1. Use the existing detector to find the face in the snapshot
        detector.process(inputImage).addOnSuccessListener { faces ->
            if (faces.isNotEmpty()) {
                val faceRect = faces[0].boundingBox
                val croppedBitmap = BitmapUtils.cropRectFromBitmap(fullFrameBitmap, faceRect)
                val standardizedFace = Bitmap.createScaledBitmap(croppedBitmap, 112, 112, false)

                // Save to disk (we use timestamp to keep filenames unique)
                saveFaceToInternalStorage(standardizedFace, fullName)

                    Toast.makeText(this, "Enrolled: $fullName", Toast.LENGTH_SHORT).show()
                } else {
                    Toast.makeText(this, "No face detected in photo! Center your face and try again.", Toast.LENGTH_LONG).show()
                }
            }
    }

    private fun saveFaceToInternalStorage(bitmap: Bitmap, folderName: String) {
        val imagesDir = File(getExternalFilesDir(null), "images")
        val studentDir = File(imagesDir, folderName)
        if (!studentDir.exists()) studentDir.mkdirs()

        val file = File(studentDir, "face_${System.currentTimeMillis()}.png")
        val out = FileOutputStream(file)
        bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
        out.close()
    }

    private fun launchQRScannerFallback() {
        // Use the deep-link URL for your AppSheet app
        val appsheetUrl = "https://www.appsheet.com/start/12a44012-61c9-466d-be5f-3f07de35e735"
        val intent = Intent(Intent.ACTION_VIEW, Uri.parse(appsheetUrl))

        // Log this for your ISO observation!
        Logger.log("Efficiency Fallback: Redirected to QR System")

        startActivity(intent)
    }

    private fun showStudentListDialog() {
        val imagesDir = File(getExternalFilesDir(null), "images")
        val studentFolders = imagesDir.listFiles()?.filter { it.isDirectory }?.map { it.name }?.toTypedArray()

        if (studentFolders.isNullOrEmpty()) {
            Toast.makeText(this, "No students enrolled yet.", Toast.LENGTH_SHORT).show()
            return
        }

        AlertDialog.Builder(this)
            .setTitle("Enrolled Students (${studentFolders.size})")
            .setItems(studentFolders) { _, which ->
                val selectedName = studentFolders[which]
                showEditDeleteOptions(selectedName) // Show options for this student
            }
            .setNegativeButton("Close", null)
            .show()
    }

    private fun showEditDeleteOptions(currentFullName: String) {
        // Add "Manage Photos" to the list
        val options = arrayOf("Add Photos", "Edit Name/Section", "Delete Student", "Cancel")

        AlertDialog.Builder(this)
            .setTitle(currentFullName)
            .setItems(options) { _, which ->
                when (which) {
                    0 -> showPhotoManagerDialog(currentFullName) // NEW FUNCTION
                    1 -> showEditDialog(currentFullName)
                    2 -> confirmDelete(currentFullName)
                }
            }.show()
    }

    private fun showPhotoManagerDialog(fullName: String) {
        val imagesDir = File(File(getExternalFilesDir(null), "images"), fullName)
        val photoFiles = imagesDir.listFiles()?.filter { it.isFile }?.toMutableList() ?: mutableListOf()

        val builder = AlertDialog.Builder(this)
        builder.setTitle("Photos for $fullName")

        // Create a list of photo names (or timestamps)
        val photoNames = photoFiles.map { it.name }.toTypedArray()

        builder.setItems(photoNames) { _, which ->
            // Option to delete a specific photo
            val fileToDelete = photoFiles[which]
            if (photoFiles.size > 1) {
                fileToDelete.delete()
                Toast.makeText(this, "Photo removed. Please restart to sync AI.", Toast.LENGTH_SHORT).show()
            } else {
                Toast.makeText(this, "Cannot delete the last photo!", Toast.LENGTH_SHORT).show()
            }
        }

        builder.setPositiveButton("Add New Photo") { _, _ ->
            val currentFrame = activityMainBinding.previewView.bitmap
            if (currentFrame != null) {
                // Use the same Auto-Cropper logic we built earlier
                registerNewStudent(currentFrame, fullName)
                Toast.makeText(this, "New angle added!", Toast.LENGTH_SHORT).show()
            }
        }
        builder.setNegativeButton("Done", null)
        builder.show()
    }

    // In MainActivity.kt, create this helper
    private fun refreshAI() {
        val images = ArrayList<Pair<String, Bitmap>>()
        val internalImagesDir = File(getExternalFilesDir(null), "images")

        internalImagesDir.listFiles()?.filter { it.isDirectory }?.forEach { folder ->
            folder.listFiles()?.forEach { file ->
                val bitmap = BitmapFactory.decodeFile(file.absolutePath)
                if (bitmap != null) {
                    images.add(Pair(folder.name, bitmap))
                }
            }
        }
        // Tell the AI to reload everything
        fileReader.run(images, fileReaderCallback)
    }

    private fun showEditDialog(oldFullName: String) {
        val dialogLayout = layoutInflater.inflate(R.layout.dialog_register, null)
        val etLast = dialogLayout.findViewById<EditText>(R.id.etLastName)
        val etFirst = dialogLayout.findViewById<EditText>(R.id.etFirstName)
        val etMI = dialogLayout.findViewById<EditText>(R.id.etMI)
        val etSection = dialogLayout.findViewById<EditText>(R.id.etSection)

        // Try to parse the old name to fill the boxes (optional but helpful)
        etLast.setText(oldFullName.substringBefore(","))

        AlertDialog.Builder(this)
            .setTitle("Edit Student Info")
            .setView(dialogLayout)
            .setPositiveButton("Update") { _, _ ->
                val newFullName = "${etLast.text.toString().uppercase()}, ${etFirst.text.toString().uppercase()} ${etMI.text.toString().uppercase()} (${etSection.text.toString().uppercase()})"

                // 1. Rename folder on disk
                val imagesDir = File(getExternalFilesDir(null), "images")
                val oldDir = File(imagesDir, oldFullName)
                val newDir = File(imagesDir, newFullName)

                if (oldDir.renameTo(newDir)) {
                    // 2. Update AI memory (faceList)
                    // This is important so the app recognizes the NEW name immediately
                    for (i in 0 until frameAnalyser.faceList.size) {
                        if (frameAnalyser.faceList[i].first == oldFullName) {
                            frameAnalyser.faceList[i] = Pair(newFullName, frameAnalyser.faceList[i].second)
                        }
                    }
                    Toast.makeText(this, "Updated successfully!", Toast.LENGTH_SHORT).show()
                }
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun confirmDelete(fullName: String) {
        AlertDialog.Builder(this)
            .setTitle("Delete Student?")
            .setMessage("This will permanently remove $fullName. Are you sure?")
            .setPositiveButton("Delete") { _, _ ->
                val dir = File(File(getExternalFilesDir(null), "images"), fullName)
                if (dir.deleteRecursively()) {
                    // Remove from AI memory
                    frameAnalyser.faceList.removeAll { it.first == fullName }
                    Toast.makeText(this, "Deleted.", Toast.LENGTH_SHORT).show()
                }
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun showDirectoryInfo() {
        // Get the current directory from the app's internal logic
        val currentDir = getExternalFilesDir(null)?.absolutePath + "/images"

        val builder = AlertDialog.Builder(this)
        builder.setTitle("Directory Settings")
        builder.setMessage("Current internal storage used:\n\n$currentDir\n\nWould you like to switch to a custom external folder?")

        builder.setPositiveButton("Change Folder") { _, _ ->
            // This triggers the original file picker from the Shubham0204 repo
            launchChooseDirectoryIntent()
        }
        builder.setNeutralButton("Use Internal Only") { _, _ ->
            // Reset to our automated internal folder
            sharedPreferences.edit().putBoolean(SHARED_PREF_IS_DATA_STORED_KEY, false).apply()
            Toast.makeText(this, "Reset to internal storage. Please restart app.", Toast.LENGTH_SHORT).show()
        }
        builder.setNegativeButton("Close", null)
        builder.show()
    }

    private fun startBurstEnrollment(fullName: String) {
        val photoCount = 10 // How many photos to take
        val interval = 500L // Time between photos in milliseconds

        lifecycleScope.launch(Dispatchers.Main) {
            Toast.makeText(this@MainActivity, "Stay still! Taking $photoCount photos...", Toast.LENGTH_SHORT).show()

            for (i in 1..photoCount) {
                val bitmap = activityMainBinding.previewView.bitmap
                if (bitmap != null) {
                    // Use the Auto-Cropper we built earlier
                    registerNewStudent(bitmap, fullName)
                    Logger.log("Burst: Captured photo $i for $fullName")
                }
                delay(interval) // Wait before taking the next one
            }

            Toast.makeText(this@MainActivity, "Enrollment Complete!", Toast.LENGTH_LONG).show()
            refreshAI() // Reload the brain with all the new photos
        }
    }

    private fun showEditUrlDialog() {
        val builder = AlertDialog.Builder(this)
        builder.setTitle("Edit Apps Script URL")

        // 1. Create an EditText to input the URL
        val input = EditText(this)
        val currentUrl = sharedPreferences.getString(PREF_URL_KEY, DEFAULT_URL)
        input.setText(currentUrl)
        builder.setView(input)

        builder.setPositiveButton("Save") { _, _ ->
            val newUrl = input.text.toString().trim()
            if (newUrl.isNotEmpty() && newUrl.contains("exec")) {
                // 2. Save it to the phone's memory
                sharedPreferences.edit().putString(PREF_URL_KEY, newUrl).apply()
                Toast.makeText(this, "URL Updated!", Toast.LENGTH_SHORT).show()
            } else {
                Toast.makeText(this, "Invalid URL format!", Toast.LENGTH_SHORT).show()
            }
        }
        builder.setNegativeButton("Cancel", null)
        builder.show()
    }

}
