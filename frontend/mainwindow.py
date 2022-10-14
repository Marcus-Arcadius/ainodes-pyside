


import pandas as pd

import backend.settings as settings
from backend.singleton import singleton
settings.load_settings_json()

gs = singleton

import random
import time, os, sys

import numpy as np
import torch
from PIL import Image
from PIL.ImageQt import ImageQt
from PySide6.QtWidgets import QProgressBar
from PySide6 import QtUiTools, QtCore
from PySide6.QtCore import QObject, QThreadPool
from PySide6.QtWidgets import QSystemTrayIcon, QListWidgetItem
from einops import rearrange

import transformers
from transformers import CLIPTokenizer, CLIPTextModel
from transformers import BertTokenizerFast

import backend.settings as settings

from backend.deforum.deforum_simplified import DeforumGenerator
from backend.ui_func import getLatestGeneratedImagesFromPath
from backend.worker import Worker

from frontend.ui_classes import *
from ldm.generate import Generate




class GenerateWindow(QObject):
    loader = QtUiTools.QUiLoader()
    file = QtCore.QFile("frontend/main/main_window.ui")
    file.open(QtCore.QFile.ReadOnly)
    w = loader.load(file)
    file.close()
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        settings.load_settings_json()

        self.image_path = ""
        self.deforum = DeforumGenerator()
        self.gr = Generate(
            weights     = 'models/sd-v1-4.ckpt',
            config     = 'configs/stable-diffusion/v1-inference.yaml',)
        gs.album = {}


        gs.models = {}
        gs.result = ""
        gs.callbackBusy = False

        gs.album = getLatestGeneratedImagesFromPath()


        self.home()

        #self.w.thumbnails.thumbs.installEventFilter(self)
        self.w.statusBar().showMessage('Ready')
        self.w.progressBar = QProgressBar()


        self.w.statusBar().addPermanentWidget(self.w.progressBar)

        # This is simply to show the bar
        self.w.progressBar.setGeometry(30, 40, 200, 25)
        self.w.progressBar.setValue(0)

        #self.nodeWindow = NodeWindow()
        self.load_history()

        self.w.actionAnim.triggered.connect(self.show_anim)
        self.w.actionPreview.triggered.connect(self.show_preview)
        self.w.actionPrompt.triggered.connect(self.show_prompt)
        self.w.actionSampler.triggered.connect(self.show_sampler)
        self.w.actionSliders.triggered.connect(self.show_sizer_count)
        self.w.actionThumbnails.triggered.connect(self.show_thumbnails)

        """self.global_factor = 1
        self.pix_map_item = QGraphicsPixmapItem()

    def scaleImage(self, factor):
        _pixmap = self.pic.scaledToHeight(int(factor*self.viewport().geometry().height()), Qt.SmoothTransformation)
        self.pix_map_item.setPixmap(_pixmap)
        self.preview.scene.setSceneRect(QRectF(_pixmap.rect()))

    def wheelEvent(self, event):
        factor = 1.5

        if QApplication.keyboardModifiers() == Qt.ControlModifier:
            view_pos = event.pos()
            scene_pos = self.mapToScene(view_pos)
            self.centerOn(scene_pos)

            if event.angleDelta().y() > 0 and self.global_factor < 20:
                self.global_factor *= factor
                self.scaleImage(self.global_factor)
            elif event.angleDelta().y() < 0 and self.global_factor > 0.2:
                self.global_factor /= factor
                self.scaleImage(self.global_factor)
        else:
            return super().wheelEvent(event)"""


    def home(self):
        self.w.thumbnails = Thumbnails()
        #self.mainPainter = QPainter()
        #self.w.initPainter(self.mainPainter)
        #self.mainPainter.begin()
        self.threadpool = QThreadPool()
        self.w.preview = Preview()
        self.w.sizer_count = SizerCount()
        self.w.sampler = Sampler()
        #self.runner = Runner()
        self.w.anim = Anim()
        self.w.prompt = Prompt()
        self.w.dynaview = Dynaview()
        self.w.dynaimage = Dynaimage()
        self.timeline = Timeline()
        self.animDials = AnimDials()
        self.animKeys = AnimKeys()

        #app2  = qapp(sys.argv)
        #self.nodes = NodeEditorWindow()
        #self.nodes.nodeeditor.addNodes()

        #wnd.show()
        self.w.thumbnails.thumbs.itemClicked.connect(self.viewImageClicked)
        self.w.thumbnails.thumbs.itemDoubleClicked.connect(self.tileImageClicked)
        #self.thumbnails.thumbs.addItem(QListWidgetItem(QIcon('frontend/main/splash.png'), "Earth"))

        self.w.sizer_count.w.heightNumber.display(str(self.w.sizer_count.w.heightSlider.value()))
        self.w.sizer_count.w.widthNumber.display(str(self.w.sizer_count.w.widthSlider.value()))
        self.w.sizer_count.w.samplesNumber.display(str(self.w.sizer_count.w.samplesSlider.value()))
        self.w.sizer_count.w.batchSizeNumber.display(str(self.w.sizer_count.w.batchSizeSlider.value()))
        self.w.sizer_count.w.stepsNumber.display(str(self.w.sizer_count.w.stepsSlider.value()))
        self.w.sizer_count.w.scaleNumber.display(str(self.w.sizer_count.w.scaleSlider.value()))

        self.w.setCentralWidget(self.w.preview.w)
        self.w.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.w.sampler.w.dockWidget)
        self.w.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.w.sizer_count.w.dockWidget)
        self.w.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.w.prompt.w.dockWidget)
        self.w.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.timeline)
        self.w.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.w.thumbnails)
        self.w.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.animDials.w.dockWidget)
        self.w.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.animKeys.w.dockWidget)

        self.w.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.w.dynaview.w.dockWidget)
        self.w.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.w.dynaimage.w.dockWidget)

        self.w.dynaview.w.setMinimumSize(QtCore.QSize(512, 512))
        #self.w.tabifyDockWidget(self.w.thumbnails, self.w.sampler.w.dockWidget)

        self.w.tabifyDockWidget( self.w.sizer_count.w.dockWidget, self.animDials.w.dockWidget)
        self.w.tabifyDockWidget(self.animDials.w.dockWidget, self.w.sampler.w.dockWidget)
        self.w.tabifyDockWidget(self.w.sampler.w.dockWidget, self.animKeys.w.dockWidget)

        self.w.tabifyDockWidget(self.w.thumbnails, self.w.dynaimage.w.dockWidget)

        self.w.tabifyDockWidget(self.timeline, self.w.prompt.w.dockWidget)

        self.w.dynaview.w.dockWidget.setMinimumSize(512, 0)





        self.animKeys.w.dockWidget.setWindowTitle('Anim Keys')
        self.w.thumbnails.setWindowTitle('Thumbnails')
        self.w.sampler.w.dockWidget.setWindowTitle('Sampler')
        self.w.sizer_count.w.dockWidget.setWindowTitle('Sliders')
        self.animDials.w.dockWidget.setWindowTitle('Dials')
        self.timeline.setWindowTitle('Timeline')
        self.w.prompt.w.dockWidget.setWindowTitle('Prompt')
        self.w.dynaview.w.dockWidget.setWindowTitle('Tensor Preview')
        self.w.dynaimage.w.dockWidget.setWindowTitle('Image Preview')
        self.w.preview.w.setWindowTitle('Canvas')
        self.vpainter = {}
        self.w.preview.w.scene = QGraphicsScene()
        self.w.preview.w.graphicsView.setScene(self.w.preview.w.scene)
        self.w.preview.canvas = QPixmap(512, 512)
        self.w.preview.canvas.fill(Qt.white)
        self.w.preview.w.scene.addPixmap(self.w.preview.canvas)
        self.w.thumbnails.thumbsZoom.valueChanged.connect(self.updateThumbsZoom)
        self.w.thumbnails.refresh.clicked.connect(self.load_history)
        self.w.imageItem = QGraphicsPixmapItem()
        #self.w.imageItem.pixmap().fill(Qt.white)
        self.newPixmap = {}
        self.tpixmap = {}
        self.updateRate = self.w.sizer_count.w.stepsSlider.value()
        self.livePainter = QPainter()
        self.vpainter["iins"] = QPainter()
        self.tpixmap = QPixmap(512, 512)
        self.ipixmap = QPixmap(512, 512)
        self.livePainter.end()
        self.vpainter["iins"].end()
    def deforumCallback(self, *args, **kwargs):
        saved_args = locals()
        #print("saved_args is", saved_args)
    def updateThumbsZoom(self):
        #with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        while gs.callbackBusy == True:
            time.sleep(0.1)
        try:
            if gs.callbackBusy == False:
                size = self.w.thumbnails.thumbsZoom.value()
                self.w.thumbnails.thumbs.setGridSize(QSize(size, size))
                self.w.thumbnails.thumbs.setIconSize(QSize(size, size))
        except Exception as e:
            print(f"Exception: {e}")
            pass
    def update_scaleNumber(self):
        float = self.w.sizer_count.w.scaleSlider.value() / 100
        self.w.sizer_count.w.scaleNumber.display(str(float))
    def update_gfpganNumber(self):
        float = self.w.sizer_count.w.gfpganSlider.value() / 10
        self.w.sizer_count.w.gfpganNumber.display(str(float))


    def show_anim(self):
        self.w.anim.w.show()
    def show_preview(self):
        self.w.preview.w.show()
    def show_prompt(self):
        self.w.prompt.w.show()
    #def show_runner(self):
    #self.runner.show()
    def show_sampler(self):
        self.w.sampler.w.show()
    def show_sizer_count(self):
        self.w.sizer_count.w.show()
    def show_thumbnails(self):
        self.w.thumbnails.show()

    def load_history(self):
        self.w.thumbnails.thumbs.clear()
        for image in gs.album:
            self.w.thumbnails.thumbs.addItem(QListWidgetItem(QIcon(image), str(image)))
    def viewThread(self, item):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            self.viewImageClicked(item)
        #worker = Worker(self.viewImageClicked(item))
        #threadpool.start(worker)
    def tileImageClicked(self, item):
        try:
            while gs.callbackBusy == True:
                time.sleep(0.1)
            #gs.callbackBusy = True
            vins = random.randint(10000, 99999)
            imageSize = item.icon().actualSize(QSize(10000, 10000))
            qimage = QImage(item.icon().pixmap(imageSize).toImage())
            self.newPixmap[vins] = QPixmap(QSize(2048, 2048))

            self.vpainter[vins] = QPainter(self.newPixmap[vins])

            newItem = QGraphicsPixmapItem()
            #vpixmap = self.w.imageItem.pixmap()


            #self.vpainter[vins].device()
            self.vpainter[vins].beginNativePainting()


            self.vpainter[vins].drawImage(QRect(QPoint(0, 0), QSize(qimage.size())), qimage)
            self.vpainter[vins].drawImage(QRect(QPoint(512, 0), QSize(qimage.size())), qimage)
            self.vpainter[vins].drawImage(QRect(QPoint(0, 512), QSize(qimage.size())), qimage)
            self.vpainter[vins].drawImage(QRect(QPoint(512, 512), QSize(qimage.size())), qimage)

            newItem.setPixmap(self.newPixmap[vins])

            #self.w.imageItem.setPixmap(vpixmap)
            #self.w.preview.w.graphicsView.modified = True
            for items in self.w.preview.w.scene.items():
                self.w.preview.w.scene.removeItem(items)
            self.w.preview.w.scene.addItem(newItem)
            self.w.preview.w.graphicsView.fitInView(newItem, Qt.AspectRatioMode.KeepAspectRatio)
            self.w.preview.w.graphicsView.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.vpainter[vins].endNativePainting()
            #gs.callbackBusy = False
        except Exception as e:
            print(f"Exception: {e}")
            pass
    def viewImageClicked(self, item):
        try:
            while gs.callbackBusy == True:
                time.sleep(0.1)
            #gs.callbackBusy = True
            vins = random.randint(10000, 99999)
            imageSize = item.icon().actualSize(QSize(10000, 10000))
            qimage = QImage(item.icon().pixmap(imageSize).toImage())
            self.newPixmap[vins] = QPixmap(qimage.size())

            self.vpainter[vins] = QPainter(self.newPixmap[vins])

            newItem = QGraphicsPixmapItem()
            #vpixmap = self.w.imageItem.pixmap()


            #self.vpainter[vins].device()
            self.vpainter[vins].beginNativePainting()


            self.vpainter[vins].drawImage(QRect(QPoint(0, 0), QSize(qimage.size())), qimage)
            newItem.setPixmap(self.newPixmap[vins])

            #self.w.imageItem.setPixmap(vpixmap)
            #self.w.preview.w.graphicsView.modified = True
            for items in self.w.preview.w.scene.items():
                self.w.preview.w.scene.removeItem(items)
            self.w.preview.w.scene.addItem(newItem)
            self.w.preview.w.graphicsView.fitInView(newItem, Qt.AspectRatioMode.KeepAspectRatio)
            self.w.preview.w.graphicsView.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.vpainter[vins].endNativePainting()
            #gs.callbackBusy = False
        except Exception as e:
            print(f"Exception: {e}")
            pass
        #self.w.preview.w.scene.update()
        #self.w.preview.w.graphicsView.setScene(self.w.preview.w.scene)

        #rad = self.w.preview.w.graphicsView.penwidth / 2 + 2
        #self.w.preview.w.graphicsView.update(QRect(self.lastPoint, position).normalized().adjusted(-rad, -rad, +rad, +rad))
        #self.w.preview.w.graphicsView.lastPoint = position







        #for item in self.w.preview.w.scene.items():
        #    self.w.preview.w.scene.removeItem(item)


        #self.w.preview.w.scene.clear()
        #imageSize = item.icon().actualSize(QSize(512, 512))
        #print(f'image item type: {type(self.w.imageItem)}')
        #self.w.imageItem.setPixmap(item.icon().pixmap(imageSize))

        #self.w.preview.w.scene.addItem(imageItem)
        #self.w.preview.w.scene.setPixmap(self.w.imageItem)

        #self.w.preview.w.scene.update()
    def taskSwitcher(self):
        choice = self.w.sampler.w.comboBox_4.currentText()
        if choice == "Text to Video":
            self.deforum_thread()
        elif choice == "Text to Image":
            self.txt2img_thread()

    def run_deforum(self, progress_callback=None):

        use_init = self.animDials.w.useInit.isChecked()
        adabins = self.animDials.w.adabins.isChecked()
        scale = self.animDials.w.scale.value()
        ddim_eta = self.animDials.w.ddim_eta.value() / 1000
        strength = self.animDials.w.strength.value() / 1000
        mask_contrast_adjust = self.animDials.w.mask_contrast.value() / 1000
        mask_brightness_adjust = self.animDials.w.mask_brightness.value() / 1000
        mask_blur = self.animDials.w.mask_blur.value() / 1000
        fov = self.animDials.w.fov.value()
        max_frames = self.animDials.w.frames.value()
        midas_weight = self.animDials.w.midas_weight.value() / 1000
        near_plane = self.animDials.w.near_plane.value()
        far_plane = self.animDials.w.far_plane.value()


        clearLatent = self.animDials.w.clearLatent.isChecked()
        clearSample = self.animDials.w.clearSample.isChecked()
        #self.init_c = None
        #if self.animDials.w.clearLatent.isChecked():
        #    self.deforum.init_latent = None
        #if self.animDials.w.clearSample.isChecked():
        #    print("clearing")
        #    self.deforum.init_c = None

        #if use_init == False:
        #    self.deforum.init_latent = None
        #    self.deforum.init_sample = None




        keyframes=self.w.prompt.w.keyFrames.toPlainText()
        prompts=self.w.prompt.w.textEdit.toPlainText()
        prompt_series = pd.Series([np.nan for a in range(max_frames)])
        if keyframes == '':
            keyframes = "0"
        prom = prompts
        key = keyframes

        new_prom = list(prom.split("\n"))
        new_key = list(key.split("\n"))

        prompts = dict(zip(new_key, new_prom))

        # for i in animation_prompts:
        #    prompt_series[i] = animation_prompts
        #    print(prompt_series[i])

        for i, prompt in prompts.items():
            n = int(i)
            prompt_series[n] = prompt
        prompt_series = prompt_series.ffill().bfill()

        #print(prompt_series)
        show_sample_per_step = True
        self.progress = 0.0
        self.update = 0
        self.steps=self.w.sizer_count.w.stepsSlider.value()

        self.onePercent = 100 / (1 * self.steps * max_frames * max_frames)
        self.updateRate = self.w.sizer_count.w.previewSlider.value()
        self.deforum.render_animation(  steps=self.steps,
                                        adabins = adabins,
                                        scale = scale,
                                        ddim_eta = ddim_eta,
                                        strength = strength,
                                        mask_contrast_adjust = mask_contrast_adjust,
                                        mask_brightness_adjust = mask_brightness_adjust,
                                        mask_blur = mask_blur,
                                        fov = fov,
                                        max_frames = max_frames,
                                        midas_weight = midas_weight,
                                        near_plane = near_plane,
                                        far_plane = far_plane,
                                        image_callback=self.image_cb,
                                        animation_prompts=prompt_series,
                                        use_init = use_init,
                                        clear_latent = clearLatent,
                                        clear_sample = clearSample,
                                        step_callback = self.deforum_step_cb,
                                        show_sample_per_step=show_sample_per_step,

                                        )
        self.stop_painters()
        self.w.thumbnails.setUpdatesEnabled(True)
    def deforum_step_cb(self, data, *args, **kwargs):
        print(data.keys())
        print(type(data['x']))
        print(type(data['i']))
        print(type(data['sigma']))
        print(type(data['sigma_hat']))
        print(type(data['denoised']))

        self.updateRate = self.w.sizer_count.w.previewSlider.value()


        self.progress = self.progress + self.onePercent
        self.w.progressBar.setValue(self.progress)
        # Pass the function to execute
        self.liveWorker2 = Worker(self.liveUpdate(data['denoised'], data['i']))

        #with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        #    self.liveUpdate(data1, data2)

        self.threadpool.start(self.liveWorker2)

    def run_txt2img(self, progress_callback=None):

        self.w.statusBar().showMessage("Loading model...")
        self.load_upscalers()


        self.updateRate = self.w.sizer_count.w.previewSlider.value()

        prompt_list = self.w.prompt.w.textEdit.toPlainText()
        prompt_list = prompt_list.split('\n')
        #self.w.setCentralWidget(self.w.dynaimage.w)
        width=self.w.sizer_count.w.widthSlider.value()
        height=self.w.sizer_count.w.heightSlider.value()
        scale=self.w.sizer_count.w.scaleSlider.value()
        self.steps=self.w.sizer_count.w.stepsSlider.value()
        samples=self.w.sizer_count.w.samplesSlider.value()
        batchsize=self.w.sizer_count.w.batchSizeSlider.value()
        seamless=self.w.sampler.w.seamless.isChecked()
        full_precision=self.w.sampler.w.fullPrecision.isChecked()
        sampler=self.w.sampler.w.comboBox.currentText()
        upscale=[self.w.sizer_count.w.upscaleSlider.value()]
        gfpgan_strength=self.w.sizer_count.w.gfpganSlider.value() / 100

        self.onePercent = 100 / (batchsize * self.steps * samples * len(prompt_list))

        if self.w.sampler.w.seedEdit.text() != '':
            seed=int(self.w.sampler.w.seedEdit.text())
        else:
            seed=''


        if gs.defaults.general.default_path_mode == "subfolders":
            outdir = gs.defaults.general.outdir
        else:
            outdir = f'{gs.defaults.general.outdir}/_batch_images'




        """The full list of arguments to Generate() are:
        gr = Generate(
                  weights     = path to model weights ('models/ldm/stable-diffusion-v1/model.ckpt')
                  config     = path to model configuraiton ('configs/stable-diffusion/v1-inference.yaml')
                  iterations  = <integer>     // how many times to run the sampling (1)
                  steps       = <integer>     // 50
                  seed        = <integer>     // current system time
                  sampler_name= ['ddim', 'k_dpm_2_a', 'k_dpm_2', 'k_euler_a', 'k_euler', 'k_heun', 'k_lms', 'plms']  // k_lms
                  grid        = <boolean>     // false
                  width       = <integer>     // image width, multiple of 64 (512)
                  height      = <integer>     // image height, multiple of 64 (512)
                  cfg_scale   = <float>       // condition-free guidance scale (7.5)
                  )

                    """

        #self.vpainter[tins] = QPainter(self.tpixmap)



        #self.vpainter[iins] = QPainter(dpixmap)

        #self.vpainter[iins].begin(dpixmap)

        self.progress = 0.0
        self.update = 0
        for i in range(batchsize):
            for prompt in prompt_list:


                results = self.gr.prompt2image(prompt   = prompt,
                                               outdir   = outdir,
                                               cfg_scale = scale,
                                               width  = width,
                                               height = height,
                                               iterations = samples,
                                               steps = self.steps,
                                               seamless = seamless,
                                               sampler_name = sampler,
                                               seed = seed,
                                               upscale = upscale,
                                               gfpgan_strength = gfpgan_strength,
                                               strength = 0.0,
                                               full_precision = full_precision,
                                               step_callback=self.testThread,
                                               image_callback=self.image_cb)
                for row in results:
                    print(f'filename={row[0]}')
                    print(f'seed    ={row[1]}')
                    filename = random.randint(10000, 99999)
                    output = f'outputs/{filename}.png'
                    row[0].save(output)
                    self.image_path = output
                    #print("We did set the image")
                    #self.w.thumbnails.thumbs.addItem(QListWidgetItem(QIcon(self.image_path), str(self.w.prompt.w.textEdit.toPlainText())))
                    #self.get_pic(clear=False)
                self.w.statusBar().showMessage("Ready...")
        self.stop_painters()
        self.w.thumbnails.setUpdatesEnabled(True)

    def deforum_thread(self):
        self.w.thumbnails.setUpdatesEnabled(False)

        worker = Worker(self.run_deforum)
        #worker.signals.progress.connect(self.testThread)
        #worker.signals.result.connect(self.stop_painters)

        # Execute
        self.threadpool.start(worker)

        #progress bar test:
        #self.progress_thread()
    def load_upscalers(self):
        gfpgan = False
        try:
            from realesrgan import RealESRGANer

            gfpgan = True
        except ModuleNotFoundError:
            pass

        if gfpgan:
            print('Loading models from RealESRGAN and facexlib')
            try:
                from basicsr.archs.rrdbnet_arch import RRDBNet
                from facexlib.utils.face_restoration_helper import FaceRestoreHelper

                RealESRGANer(
                    scale=2,
                    model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                    model=RRDBNet(
                        num_in_ch=3,
                        num_out_ch=3,
                        num_feat=64,
                        num_block=23,
                        num_grow_ch=32,
                        scale=2,
                    ),
                )

                RealESRGANer(
                    scale=4,
                    model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
                    model=RRDBNet(
                        num_in_ch=3,
                        num_out_ch=3,
                        num_feat=64,
                        num_block=23,
                        num_grow_ch=32,
                        scale=4,
                    ),
                )

                FaceRestoreHelper(1, det_model='retinaface_resnet50')
                print('...success')
            except Exception:
                import traceback

                print('Error loading GFPGAN:')
                print(traceback.format_exc())
    def prepare_loading(self):
        transformers.logging.set_verbosity_error()

        # this will preload the Bert tokenizer fles
        print('preloading bert tokenizer...')

        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        print('...success')

        # this will download requirements for Kornia
        print('preloading Kornia requirements (ignore the deprecation warnings)...')
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            import kornia
        print('...success')

        version = 'openai/clip-vit-large-patch14'

        print('preloading CLIP model (Ignore the deprecation warnings)...')
        sys.stdout.flush()
        self.load_upscalers()
        tokenizer = CLIPTokenizer.from_pretrained(version)
        transformer = CLIPTextModel.from_pretrained(version)
        print('\n\n...success')

        # In the event that the user has installed GFPGAN and also elected to use
        # RealESRGAN, this will attempt to download the model needed by RealESRGANer




    def stop_painters(self):
        try:
            self.vpainter["iins"].end()
            self.livePainter.end()
        except Exception as e:
            print(f"Exception: {e}")
            pass
    def txt2img_thread(self):
        self.w.thumbnails.setUpdatesEnabled(False)
        #self.run_txt2img()
        # Pass the function to execute
        worker = Worker(self.run_txt2img)
        #worker.signals.progress.connect(self.testThread)
        #worker.signals.result.connect(self.stop_painters)

        # Execute
        self.threadpool.start(worker)

        #progress bar test:
        #self.progress_thread()
    def testThread(self, data1=None, data2=None):


        self.updateRate = self.w.sizer_count.w.previewSlider.value()


        self.progress = self.progress + self.onePercent
        self.w.progressBar.setValue(self.progress)
        # Pass the function to execute
        self.liveWorker2 = Worker(self.liveUpdate(data1, data2))

        #with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        #    self.liveUpdate(data1, data2)

        self.threadpool.start(self.liveWorker2)

    def liveUpdate(self, data1=None, data2=None):
        self.w.statusBar().showMessage(f"Generating... (step {data2} of {self.steps})")

        if self.update >= self.updateRate:
            try:
                self.update = 0

                #QtConcurrent.run(self.threadpool, self.test_output, data1, data2)
                self.test_output(data1, data2)
            except Exception as e:
                print(f"Exception: {e}")
                pass
                self.update = 0
        else:
            self.update += 1
        return self.pass_object


    def test_output(self, data1, data2):

        if gs.callbackBusy == False:
            try:
                self.livePainter.begin(self.tpixmap)
                gs.callbackBusy = True

                #transform = T.ToPILImage()
                #img = transform(data1)
                #img = Image.fromarray(data1.astype(np.uint8))
                #img = QImage.fromTensor(data1)

                x_samples = torch.clamp((data1 + 1.0) / 2.0, min=0.0, max=1.0)
                if len(x_samples) != 1:
                    raise Exception(
                        f'>> expected to get a single image, but got {len(x_samples)}')
                x_sample = 255.0 * rearrange(
                    x_samples[0].cpu().numpy(), 'c h w -> h w c'
                )

                #self.x_sample = cv2.cvtColor(self.x_sample.astype(np.uint8), cv2.COLOR_RGB2BGR)
                x_sample = x_sample.astype(np.uint8)
                dPILimg = Image.fromarray(x_sample)

                tins = random.randint(10000, 99999)

                #self.tpixmap = QPixmap(512, 512)
                #self.vpainter[tins] = QPainter(self.tpixmap)

                #self.vpainter[tins].begin(self.tpixmap)
                #self.vpainter[tins].device()
                dqimg = ImageQt(dPILimg)
                #self.qimage[tins] = ImageQt(dPILimg)
                self.livePainter.drawImage(QRect(0, 0, 512, 512), dqimg)
                self.w.dynaview.w.label.setPixmap(self.tpixmap.scaled(512, 512, Qt.AspectRatioMode.KeepAspectRatio))
                #self.w.dynaview.w.label.setPixmap(self.tpixmap[tins].scaled(512, 512, Qt.AspectRatioMode.IgnoreAspectRatio))
                #self.vpainter[tins].end()
                #self.w.dynaview.w.label.update()
                #gs.callbackBusy = False

                #dqimg = ImageQt(dPILimg)
                #qimg = QPixmap.fromImage(dqimg)
                self.livePainter.end()
                #self.vpainter[tins].end()


                gs.callbackBusy = False
                #result = data2
                return self.pass_object
            except Exception as e:
                print(f"Exception: {e}")
                return self.pass_object
        #dynapixmap = QPixmap(QPixmap.fromImage(dqimg))
    def pass_object(self, progress_callback=None):
        pass
    def image_cb(self, image, seed=None, upscaled=False, use_prefix=None, first_seed=None):
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:

            try:

                #gs.callbackBusy = True
                #dimg = ImageQt(image)
                #dpixmap = QPixmap(QPixmap.fromImage(dimg))
                #iins = random.randint(10000, 99999)
                #dpixmap = QPixmap(512, 512)

                #self.vpainter[iins].device()


                try:
                    self.vpainter["iins"].begin(self.ipixmap)
                except:
                    pass
                qimage = ImageQt(image)
                self.vpainter["iins"].drawImage(QRect(0, 0, 512, 512), qimage)



                self.w.dynaimage.w.label.setPixmap(self.ipixmap.scaled(512, 512, Qt.AspectRatioMode.KeepAspectRatio))
                #self.vpainter["iins"].end()
                #gs.callbackBusy = False
                #self.w.dynaimage.w.label.update()
                self.vpainter["iins"].end()
            except:
                pass

    def get_pic(self, clear=False): #from self.image_path
        #for item in self.w.preview.w.scene.items():
        #    self.w.preview.w.scene.removeItem(item)

        image_qt = QImage(self.image_path)

        self.w.preview.pic = QGraphicsPixmapItem()
        self.w.preview.pic.setPixmap(QPixmap.fromImage(image_qt))
        if clear == True:
            self.w.preview.w.scene.clear()
        self.w.preview.w.scene.addItem(self.w.preview.pic)

        self.w.preview.w.graphicsView.fitInView(self.w.preview.pic, Qt.AspectRatioMode.KeepAspectRatio)
        self.w.preview.w.graphicsView.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        #gs.obj_to_delete = self.w.preview.pic
    def zoom_IN(self):
        self.w.preview.w.graphicsView.scale(1.25, 1.25)
    def zoom_OUT(self):
        self.w.preview.w.graphicsView.scale(0.75, 0.75)


    def eventFilter(self, source, event):
        if event.type() == QEvent.ContextMenu and source is self.w.thumbnails.thumbs:
            menu = QMenu()
            tileAction = QAction("Tile", self)
            menu.addAction(tileAction)
            menu.addAction('Action 2')
            menu.addAction('Action 3')

            if menu.exec_(event.globalPos()):
                item = source.itemAt(event.pos())
                self.tileImageClicked(item)
                #print(item.text())
            return True
        return super().eventFilter(source, event)


"""class CalculatorWin(CalculatorWindow):
    def __init__(self, *args, **kwargs):




        app2 = qapp(sys.argv)
        nodes = CalculatorWindow()
        nodes.show()
        app2.exec()"""
"""def show_nodes():
    #in main thread:
    CalculatorWin()

    #in a separate thread
    #worker = Worker(CalculatorWin) # Any other args, kwargs are passed to the run function
    # Execute
    #threadpool.start(worker)"""





'''
Deforum Generator Class is used to generate Animations with Stable Diffusion
to use it, just call:

mp4 = DeforumGenerator(gs) (makevideo implementation needed)

It also takes the following parameters:


    prompts = [
        "a beautiful forest by Asher Brown Durand, trending on Artstation",  # the first prompt I want
        "a beautiful portrait of a woman by Artgerm, trending on Artstation",  # the second prompt I want
        # "this prompt I don't want it I commented it out",
        # "a nousr robot, trending on Artstation", # use "nousr robot" with the robot diffusion model (see model_checkpoint setting)
        # "touhou 1girl komeiji_koishi portrait, green hair", # waifu diffusion prompts can use danbooru tag groups (see model_checkpoint)
        # "this prompt has weights if prompt weighting enabled:2 can also do negative:-2", # (see prompt_weighting)
    ]
    animation_prompts = {
        0: "The perfect symmetric violin",
        20: "the perfect symmetric violin",
        30: "a beautiful coconut, trending on Artstation",
        40: "a beautiful durian, trending on Artstation",
    }
    W = 512  # @param
    H = 512  # @param
    W, H = map(lambda x: x - x % 64, (W, H))  # resize to integer multiple of 64
    seed = -1  # @param
    sampler = 'klms'  # @param ["klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral","plms", "ddim"]
    steps = 20  # @param
    scale = 7  # @param
    ddim_eta = 0.0  # @param
    dynamic_threshold = None
    static_threshold = None
    # @markdown **Save & Display Settings**
    save_samples = True  # @param {type:"boolean"}
    save_settings = True  # @param {type:"boolean"}
    display_samples = True  # @param {type:"boolean"}
    save_sample_per_step = False  # @param {type:"boolean"}
    show_sample_per_step = False  # @param {type:"boolean"}
    prompt_weighting = False  # @param {type:"boolean"}
    normalize_prompt_weights = True  # @param {type:"boolean"}
    log_weighted_subprompts = False  # @param {type:"boolean"}

    n_batch = 1  # @param
    batch_name = "StableFun"  # @param {type:"string"}
    filename_format = "{timestring}_{index}_{prompt}.png"  # @param ["{timestring}_{index}_{seed}.png","{timestring}_{index}_{prompt}.png"]
    seed_behavior = "iter"  # @param ["iter","fixed","random"]
    make_grid = False  # @param {type:"boolean"}
    grid_rows = 2  # @param
    outdir = "output"
    use_init = False  # @param {type:"boolean"}
    strength = 0.0  # @param {type:"number"}
    strength_0_no_init = True  # Set the strength to 0 automatically when no init image is used
    init_image = "https://cdn.pixabay.com/photo/2022/07/30/13/10/green-longhorn-beetle-7353749_1280.jpg"  # @param {type:"string"}
    # Whiter areas of the mask are areas that change more
    use_mask = False  # @param {type:"boolean"}
    use_alpha_as_mask = False  # use the alpha channel of the init image as the mask
    mask_file = "https://www.filterforge.com/wiki/images/archive/b/b7/20080927223728%21Polygonal_gradient_thumb.jpg"  # @param {type:"string"}
    invert_mask = False  # @param {type:"boolean"}
    # Adjust mask image, 1.0 is no adjustment. Should be positive numbers.
    mask_brightness_adjust = 1.0  # @param {type:"number"}
    mask_contrast_adjust = 1.0  # @param {type:"number"}
    # Overlay the masked image at the end of the generation so it does not get degraded by encoding and decoding
    overlay_mask = True  # {type:"boolean"}
    # Blur edges of final overlay mask, if used. Minimum = 0 (no blur)
    mask_overlay_blur = 5  # {type:"number"}

    n_samples = 1  # doesnt do anything
    precision = 'autocast'
    C = 4
    f = 8

    prompt = ""
    timestring = ""
    init_latent = None
    init_sample = None
    init_c = None

    # Anim Args

    animation_mode = '3D'  # @param ['None', '2D', '3D', 'Video Input', 'Interpolation'] {type:'string'}
    max_frames = 10  # @Dparam {type:"number"}
    border = 'replicate'  # @param ['wrap', 'replicate'] {type:'string'}
    # @markdown ####**Motion Parameters:**
    angle = "0:(0)"  # @param {type:"string"}
    zoom = "0:(1.04)"  # @param {type:"string"}
    translation_x = "0:(10*sin(2*3.14*t/10))"  # @param {type:"string"}
    translation_y = "0:(0)"  # @param {type:"string"}
    translation_z = "0:(10)"  # @param {type:"string"}
    rotation_3d_x = "0:(0)"  # @param {type:"string"}
    rotation_3d_y = "0:(0)"  # @param {type:"string"}
    rotation_3d_z = "0:(0)"  # @param {type:"string"}
    flip_2d_perspective = False  # @param {type:"boolean"}
    perspective_flip_theta = "0:(0)"  # @param {type:"string"}
    perspective_flip_phi = "0:(t%15)"  # @param {type:"string"}
    perspective_flip_gamma = "0:(0)"  # @param {type:"string"}
    perspective_flip_fv = "0:(53)"  # @param {type:"string"}
    noise_schedule = "0: (0.02)"  # @param {type:"string"}
    strength_schedule = "0: (0.65)"  # @param {type:"string"}
    contrast_schedule = "0: (1.0)"  # @param {type:"string"}
    # @markdown ####**Coherence:**
    color_coherence = 'Match Frame 0 LAB'  # @param ['None', 'Match Frame 0 HSV', 'Match Frame 0 LAB', 'Match Frame 0 RGB'] {type:'string'}
    diffusion_cadence = '1'  # @param ['1','2','3','4','5','6','7','8'] {type:'string'}
    # @markdown ####**3D Depth Warping:**
    use_depth_warping = True  # @param {type:"boolean"}
    midas_weight = 0.3  # @param {type:"number"}
    near_plane = 200
    far_plane = 10000
    fov = 40  # @param {type:"number"}
    padding_mode = 'border'  # @param ['border', 'reflection', 'zeros'] {type:'string'}
    sampling_mode = 'bicubic'  # @param ['bicubic', 'bilinear', 'nearest'] {type:'string'}
    save_depth_maps = False  # @param {type:"boolean"}

    # @markdown ####**Video Input:**
    video_init_path = '/content/video_in.mp4'  # @param {type:"string"}
    extract_nth_frame = 1  # @param {type:"number"}
    overwrite_extracted_frames = True  # @param {type:"boolean"}
    use_mask_video = False  # @param {type:"boolean"}
    video_mask_path = '/content/video_in.mp4'  # @param {type:"string"}

    # @markdown ####**Interpolation:**
    interpolate_key_frames = False  # @param {type:"boolean"}
    interpolate_x_frames = 4  # @param {type:"number"}

    # @markdown ####**Resume Animation:**
    resume_from_timestring = False  # @param {type:"boolean"}
    resume_timestring = "20220829210106"  # @param {type:"string"}
'''


