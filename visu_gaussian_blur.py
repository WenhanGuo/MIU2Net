import numpy as np
import scipy.ndimage as ndimage
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox

path = '/Users/danny/Desktop/WL/data_new/kappa/map_24012.fits'
std_init = 1   # initialize gaussian blur std slidebar

# open fits and apply initial gaussian blur
img = fits.open(path)[0].data * 100
img = np.expand_dims(np.float32(img), 0)   # simulate torch tensor shape of kappa
img = img[0]
img_blur = ndimage.gaussian_filter(img, sigma=std_init, order=0)

# plot original kappa & blurred kappa side by side
fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2)
im1 = ax1.imshow(img, cmap=plt.cm.jet, vmin=-3, vmax=7)
ax1.set_title('True kappa')
im2 = ax2.imshow(img_blur, cmap=plt.cm.jet, vmin=-3, vmax=7)
ax2.set_title('Blurred kappa')
cbar_ax = fig.add_axes([0.85, 0.25, 0.02, 0.6])
fig.colorbar(im1, cax=cbar_ax)

# add slidebar for gaussian blur standard deviation
ax_sigma = plt.axes([0.2, 0.15, 0.65, 0.03])
sigma_slider = Slider(ax=ax_sigma, label='standard deviation', valmin=0.1, valmax=50, valinit=std_init)
ax_sigma_text = plt.axes([0.85, 0.15, 0.1, 0.03])
sigma_text_box = TextBox(ax=ax_sigma_text, label='', initial=str(std_init))

# padding plt fig to leave room for colorbar & slidebar
fig.subplots_adjust(right=0.8, bottom=0.2)

def update_slider(val):
    sigma = sigma_slider.val
    img_blur = ndimage.gaussian_filter(img, sigma=sigma, order=0)
    im2.set_data(img_blur)

    sigma_text_box.set_val(str(sigma))
    fig.canvas.draw_idle()

sigma_slider.on_changed(update_slider)

def update_text_box(text):
    try:
        sigma = float(sigma_text_box.text)
        sigma_slider.set_val(round(sigma,2))
    except:
        pass
    fig.canvas.draw_idle()

sigma_text_box.on_submit(update_text_box)

plt.show()