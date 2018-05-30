# Written by David McDougall, 2018

""" From Wikipedia "Retina": Although there are more than 130 million
retinal receptors, there are only approximately 1.2 million fibres
(axons) in the optic nerve; a large amount of pre-processing is
performed within the retina. The fovea produces the most accurate
information. Despite occupying about 0.01% of the visual field (less
than 2 of visual angle), about 10% of axons in the optic nerve are
devoted to the fovea.

Fun Fact 1: The human optic nerve has 800,000 ~ 1,700,000 nerve fibers.
Fun Fact 2: The human eye can distiguish between 10 million different colors.
Sources: Wikipedia. """

import numpy as np
import cv2
import scipy.misc
import math
import scipy.ndimage
import random
import PIL, PIL.ImageDraw
import matplotlib.pyplot as plt
from sdr import SDR
import encoders


class Eye:
    """
    Optic sensor with central fovae.

    Attribute output_sdr ... retina's output
    Attribute roi ... The most recent view, kept as a attribute.
    Attribute parvo ... 
    Attribute magno ... 

    The following three attributes control where the eye is looking within
    the image.  They are Read/Writable.
    Attribute position     (X, Y) coords of eye center within image
    Attribute orientation  ... units are radians
    Attribute scale        ... 
    """
    def __init__(self,
        output_diameter   = 200,
        resolution_factor = 3,
        fovea_scale       = .177,
        sparsity          = .2,):
        """
        Argument output_diameter is size of output ...
        Argument resolution_factor is used to expand the sensor array so that
            the fovea has adequate resolution.  After log-polar transform image
            is reduced by this factor back to the output_diameter.
        Argument fovea_scale is magic number ...
        Argument sparsity is fraction of bits in eye.output_sdr which are 
            active, on average.
        """
        self.output_diameter   = output_diameter
        self.retina_diameter   = int(resolution_factor * output_diameter)
        self.fovea_scale       = fovea_scale
        assert(output_diameter // 2 * 2 == output_diameter) # Diameter must be an even number.
        assert(self.retina_diameter // 2 * 2 == self.retina_diameter) # (Resolution Factor X Diameter) must be an even number.

        self.output_sdr = SDR((output_diameter, output_diameter, 2,))

        self.retina = cv2.bioinspired.Retina_create(
            inputSize            = (self.retina_diameter, self.retina_diameter),
            colorMode            = True,
            colorSamplingMethod  = cv2.bioinspired.RETINA_COLOR_BAYER,)

        print(self.retina.printSetup())
        print()

        self.parvo_enc = encoders.ChannelEncoder(
                            input_shape = (output_diameter, output_diameter, 3,),
                            num_samples = 1, sparsity = sparsity ** (1/3.),
                            dtype=np.uint8, drange=[0, 255,])

        self.magno_enc = encoders.ChannelEncoder(
                            input_shape = (output_diameter, output_diameter),
                            num_samples = 1, sparsity = sparsity,
                            dtype=np.uint8, drange=[0, 255],)

        self.image_file = None
        self.image = None

    def new_image(self, image):
        """
        Argument image ...
            If String, will load image from file path.
            If numpy.ndarray, will attempt to cast to correct data type and
                dimensions.
        """
        # Load image if needed.
        if isinstance(image, str):
            self.image_file = image
            self.image = np.array(PIL.Image.open(image), copy=False)
        else:
            self.image_file = None
            self.image = image
        # Get the image into the right format.
        assert(isinstance(self.image, np.ndarray))
        if self.image.dtype != np.uint8:
            raise TypeError('Image "%s" dtype is not unsigned 8 bit integer, image.dtype is %s.'%(
                self.image_file if self.image_file is not None else 'argument',
                self.image.dtype))
        # Ensure there are three color channels.
        if len(self.image.shape) == 2 or self.image.shape[2] == 1:
            self.image = np.dstack([self.image] * 3)
        # Drop the alpha channel if present.
        elif self.image.shape[2] == 4:
            self.image = self.image[:,:,:3]
        # Sanity checks.
        assert(len(self.image.shape) == 3)
        assert(self.image.shape[2] == 3) # Color images only.

        self.reset()
        self.center_view()

    def center_view(self):
        """Center the view over the image"""
        self.orientation = 0
        self.position    = (self.image.shape[0]/2., self.image.shape[1]/2.)
        self.scale       = np.min(np.divide(self.image.shape[:2], self.retina_diameter))

    def randomize_view(self, scale_range=None):
        """Set the eye's view point to a random location"""
        if scale_range is None:
            scale_range = [2, min(self.image.shape[:2]) / self.retina_diameter]
        self.orientation = random.uniform(0, 2 * math.pi)
        self.scale       = random.uniform(min(scale_range), max(scale_range))
        roi_radius       = self.scale * self.retina_diameter / 2
        self.position    = [random.uniform(roi_radius, dim - roi_radius)
                                 for dim in self.image.shape[:2]]

    def _crop_roi(self):
        """
        Crop to Region Of Interest (ROI) which contains the whole field of view.
        Note that the size of the ROI is (eye.output_diameter *
        eye.resolution_factor).

        Arguments: eye.scale, eye.position, eye.image

        Returns RGB image.
        """
        r     = int(round(self.scale * self.retina_diameter / 2))
        x, y  = self.position
        x     = int(round(x))
        y     = int(round(y))
        x_max, y_max, color_depth = self.image.shape
        # Find the boundary of the ROI and slice out the image.
        x_low  = max(0, x-r)
        x_high = min(x_max, x+r)
        y_low  = max(0, y-r)
        y_high = min(y_max, y+r)
        image_slice = self.image[x_low : x_high, y_low : y_high]
        # Make the ROI and insert the image into it.
        roi = np.zeros((2*r, 2*r, 3,), dtype=np.uint8)
        if x-r < 0:
            x_offset = abs(x-r)
        else:
            x_offset = 0
        if y-r < 0:
            y_offset = abs(y-r)
        else:
            y_offset = 0
        x_shape, y_shape, color_depth = image_slice.shape
        roi[x_offset:x_offset+x_shape, y_offset:y_offset+y_shape] = image_slice
        # Rescale the ROI to remove the scaling effect.
        roi = scipy.misc.imresize(roi, (self.retina_diameter, self.retina_diameter))
        return roi

    def compute(self):
        self.roi = self._crop_roi()

        # Retina image transforms (Parvo & Magnocellular).
        self.retina.run(self.roi)
        parvo = self.retina.getParvo()
        magno = self.retina.getMagno()

        # Log Polar Transform.
        center = self.retina_diameter / 2
        M      = self.retina_diameter * self.fovea_scale
        parvo = cv2.logPolar(parvo,
            center = (center, center),
            M      = M,
            flags  = cv2.WARP_FILL_OUTLIERS)
        magno = cv2.logPolar(magno,
            center = (center, center),
            M      = M,
            flags  = cv2.WARP_FILL_OUTLIERS)
        parvo = scipy.misc.imresize(parvo, (self.output_diameter, self.output_diameter))
        magno = scipy.misc.imresize(magno, (self.output_diameter, self.output_diameter))

        # Apply rotation by rolling the images around axis 1.
        rotation = self.output_diameter * self.orientation / (2 * math.pi)
        rotation = int(round(rotation))
        self.parvo = np.roll(parvo, rotation, axis=0)
        self.magno = np.roll(magno, rotation, axis=0)

        # Encode images into SDRs.
        p   = self.parvo_enc.encode(self.parvo)
        pr, pg, pb = np.dsplit(p, 3)
        p   = np.logical_and(np.logical_and(pr, pg), pb)
        p   = np.expand_dims(np.squeeze(p), axis=2)
        m   = self.magno_enc.encode(self.magno)
        sdr = np.concatenate([p, m], axis=2)
        self.output_sdr.dense = sdr
        return self.output_sdr

    def make_roi_pretty(self, roi=None):
        """
        Makes the eye's view look more presentable.
        - Adds a black circular boarder to mask out areas which the eye can't see
          Note that this boarder is actually a bit too far out, playing with
          eye.fovea_scale can hide areas which this ROI image will show.
        - Adds 5 dots to the center of the image to show where the fovea is.

        Returns an RGB image.
        """
        if roi is None:
            roi = self.roi

        # Show the ROI, first rotate it like the eye is rotated.
        angle = self.orientation * 360 / (2 * math.pi)
        roi = self.roi[:,:,::-1]
        rows, cols, color_depth = roi.shape
        M   = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        roi = cv2.warpAffine(roi, M, (cols,rows))

        # Mask out areas the eye can't see by drawing a circle boarder.
        center = int(roi.shape[0] / 2)
        circle_mask = np.zeros(roi.shape, dtype=np.uint8)
        cv2.circle(circle_mask, (center, center), center, thickness = -1, color=(255,255,255))
        roi = np.minimum(roi, circle_mask)

        # Invert 5 pixels in the center to show where the fovea is located.
        roi[center, center]     = np.full(3, 255) - roi[center, center]
        roi[center+2, center+2] = np.full(3, 255) - roi[center+2, center+2]
        roi[center-2, center+2] = np.full(3, 255) - roi[center-2, center+2]
        roi[center-2, center-2] = np.full(3, 255) - roi[center-2, center-2]
        roi[center+2, center-2] = np.full(3, 255) - roi[center+2, center-2]
        return roi

    def show_view(self, window_name='Eye'):
        if False:
            print("Sparsity %g"%(len(self.output_sdr) / self.output_sdr.size))
            parvo = self.output_sdr.dense[:,:,0]
            magno = self.output_sdr.dense[:,:,1]
            print("Parvo Sparsity %g"%(np.count_nonzero(parvo) / np.product(parvo.shape)))
            print("Magno Sparsity %g"%(np.count_nonzero(magno) / np.product(magno.shape)))
        roi = self.make_roi_pretty()
        cv2.imshow('Region Of Interest', roi)
        cv2.imshow('Parvocellular', self.parvo[:,:,::-1])
        cv2.imshow('Magnocellular', self.magno)
        cv2.waitKey(1)

    def input_space_sample_points(self, npoints):
        """
        Returns a sampling of coordinates which the eye is currently looking at.
        Use the result to determine the actual label of the image in the area
        where the eye is looking.
        """
        # Find the retina's radius in the image.
        r = int(round(self.scale * self.retina_diameter / 2))
        # Shrink the retina's radius so that sample points are nearer the fovea.
        # Also shrink radius B/C this does not account for the diagonal
        # distance, just the manhattan distance.
        r = r * 2/3
        # Generate points.
        coords = np.random.random_integers(-r, r, size=(npoints, 2))
        # Add this position offset.
        coords += np.array(np.rint(self.position), dtype=np.int).reshape(1, 2)
        return coords

    def reset(self):
        self.retina.clearBuffers()


class EyeSensorSampler:
    """
    Samples eyesensor.rgb, the eye's view.

    Attribute samples is list of RGB numpy arrays.
    """
    def __init__(self, eyesensor, sample_period, number_of_samples=30):
        """
        This draws its samples directly from the output of eyesensor.view() by
        wrapping the method.
        """
        self.sensor         = sensor = eyesensor
        self.sensor_compute = sensor.compute
        self.sensor.compute = self.compute
        self.age          = 0
        self.samples      = []
        number_of_samples = min(number_of_samples, sample_period)   # Don't die.
        self.schedule     = random.sample(range(sample_period), number_of_samples)
        self.schedule.sort(reverse=True)

    def compute(self, *args, **kw_args):
        """Wrapper around eyesensor.view which takes samples"""
        retval = self.sensor_compute(*args, **kw_args)
        if self.schedule and self.age == self.schedule[-1]:
            self.schedule.pop()
            roi = self.sensor.make_roi_pretty(self.sensor.roi)
            self.samples.append(roi)
        self.age += 1
        return retval

    def view_samples(self, show=True):
        """Displays the samples."""
        if not self.samples:
            return  # Nothing to show...
        plt.figure("Sample views")
        num = len(self.samples)
        rows = math.floor(num ** .5)
        cols = math.ceil(num / rows)
        for idx, img in enumerate(self.samples):
            plt.subplot(rows, cols, idx+1)
            plt.imshow(img[:,:,::-1], interpolation='nearest')
        if show:
            plt.show()


# TODO: Consider splitting motor controls and motor sensory into different
# classes...
#
#
# EXPERIMENT: Try breaking out each output encoder by type instead of
# concatenating them all together.  Each type of sensors would then get its own
# HTM.  Maybe keep the derivatives with their source?
#
class EyeController:
    """
    Motor controller for the EyeSensor class.

    The eye sensor has 4 degrees of freedom: X and Y location, scale, and
    orientation. These values can be controlled by activating control vectors,
    each of which  has a small but cumulative effect.  CV's are normally
    distributed with a mean of zero.  Activate control vectors by calling
    controller.move(control-vectors).

    The controller outputs its current location, scale and orientation as well
    as their first derivatives w/r/t time as an SDR.
    """
    def __init__(self, eye_sensor,
        # Control Vector Parameters
        num_cv                      = 600,
        pos_stddev                  = 1,
        angle_stddev                = math.pi / 8,
        scale_stddev                = 2,
        # Motor Sensor Parameters
        position_encoder            = None,
        velocity_encoder            = None,
        angle_encoder               = None,
        angular_velocity_encoder    = None,
        scale_encoder               = None,
        scale_velocity_encoder      = None,):
        """
        Argument num_cv is the approximate number of control vectors to use.
        Arguments pos_stddev, angle_stddev, and scale_stddev are the standard
                  deviations of the control vector movements, control vectors
                  are normally distributed about a mean of 0.

        Arguments position_encoder, velocity_encoder, angle_encoder,
                  angular_velocity_encoder, scale_encoder, and
                  scale_velocity_encoder are instances of
                  RandomDistributedScalarEncoderParameters.

        Attribute control_sdr ... eye movement input controls
        Attribute motor_sdr ... internal motor sensor output

        Attribute gaze is a list of tuples of (X, Y, Orientation, Scale)
                  History of recent movements, self.move() updates this.
                  This is cleared by the following methods:
                      self.new_image() 
                      self.center_view()
                      self.randomize_view()
        """
        assert(isinstance(parameters, EyeControllerParameters))
        assert(isinstance(eye_sensor, EyeSensor))
        self.args = args = parameters
        self.eye_sensor  = eye_sensor
        self.control_vectors, self.control_sdr = self.make_control_vectors(
                num_cv       = args.num_cv,
                pos_stddev   = args.pos_stddev,
                angle_stddev = args.angle_stddev,
                scale_stddev = args.scale_stddev,)

        self.motor_position_encoder         = RandomDistributedScalarEncoder(args.position_encoder)
        self.motor_angle_encoder            = RandomDistributedScalarEncoder(args.angle_encoder)
        self.motor_scale_encoder            = RandomDistributedScalarEncoder(args.scale_encoder)
        self.motor_velocity_encoder         = RandomDistributedScalarEncoder(args.velocity_encoder)
        self.motor_angular_velocity_encoder = RandomDistributedScalarEncoder(args.angular_velocity_encoder)
        self.motor_scale_velocity_encoder   = RandomDistributedScalarEncoder(args.scale_velocity_encoder)
        self.motor_encoders = [ self.motor_position_encoder,    # X Posititon
                                self.motor_position_encoder,    # Y Position
                                self.motor_angle_encoder,
                                self.motor_scale_encoder,
                                self.motor_velocity_encoder,    # X Velocity
                                self.motor_velocity_encoder,    # Y Velocity
                                self.motor_angular_velocity_encoder,
                                self.motor_scale_velocity_encoder,]
        self.motor_sdr = SDR((sum(enc.output.size for enc in self.motor_encoders),))
        self.gaze = []

    @staticmethod
    def make_control_vectors(num_cv, pos_stddev, angle_stddev, scale_stddev):
        """
        Argument num_cv is the approximate number of control vectors to create
        Arguments pos_stddev, angle_stddev, and scale_stddev are the standard
                  deviations of the controls effects of position, angle, and 
                  scale.

        Returns pair of control_vectors, control_sdr

        The control_vectors determines what happens for each output. Each
        control is a 4-tuple of (X, Y, Angle, Scale) movements. To move,
        active controls are summed and applied to the current location.
        control_sdr contains the shape of the control_vectors.
        """
        cv_sz = int(round(num_cv // 6))
        control_shape = (6*cv_sz,)

        pos_controls = [
            (random.gauss(0, pos_stddev), random.gauss(0, pos_stddev), 0, 0)
                for i in range(4*cv_sz)]

        angle_controls = [
            (0, 0, random.gauss(0, angle_stddev), 0)
                for angle_control in range(cv_sz)]

        scale_controls = [
            (0, 0, 0, random.gauss(0, scale_stddev))
                for scale_control in range(cv_sz)]

        control_vectors = pos_controls + angle_controls + scale_controls
        random.shuffle(control_vectors)
        control_vectors = np.array(control_vectors)

        # Add a little noise to all control vectors
        control_vectors[:, 0] += np.random.normal(0, pos_stddev/10,    control_shape)
        control_vectors[:, 1] += np.random.normal(0, pos_stddev/10,    control_shape)
        control_vectors[:, 2] += np.random.normal(0, angle_stddev/10,  control_shape)
        control_vectors[:, 3] += np.random.normal(0, scale_stddev/10,  control_shape)
        return control_vectors, SDR(control_shape)

    def move(self, control_sdr=None, min_dist_from_edge=0):
        """
        Apply the given controls to the current gaze location and updates the
        motor sdr accordingly.

        Argument control_sdr is assigned into this classes attribute
                 self.control_sdr.  It represents the control vectors to use.
                 The selected control vectors are summed and their effect is
                 applied to the eye's location.

        Returns an SDR encoded representation of the eyes new location and 
        velocity.
        """
        self.control_sdr.assign(control_sdr)
        eye = self.eye_sensor
        # Calculate the forces on the motor
        controls = self.control_vectors[self.control_sdr.index]
        controls = np.sum(controls, axis=0)
        dx, dy, dangle, dscale = controls
        # Calculate the new rotation
        eye.orientation = (eye.orientation + dangle) % (2*math.pi)
        # Calculate the new scale
        new_scale  = np.clip(eye.scale + dscale, eye.args.min_scale, eye.args.max_scale)
        real_ds    = new_scale - eye.scale
        avg_scale  = (new_scale + eye.scale) / 2
        eye.scale = new_scale
        # Scale the movement such that the same CV yields the same visual
        # displacement, regardless of scale.
        dx       *= avg_scale
        dy       *= avg_scale
        # Calculate the new position.  
        x, y     = eye.position
        p        = [x + dx, y + dy]
        edge     = min_dist_from_edge
        p        = np.clip(p, [edge,edge], np.subtract(eye.image.shape[:2], edge))
        real_dp  = np.subtract(p, eye.position)
        eye.position = p
        # Book keeping.
        self.gaze.append(tuple(eye.position) + (eye.orientation, eye.scale))
        # Put together information about the motor.
        velocity = (
            eye.position[0],
            eye.position[1],
            eye.orientation,
            eye.scale,
            real_dp[0],
            real_dp[1],
            dangle,
            real_ds,
        )
        # Encode the motors sensors and concatenate them into one big SDR.
        v_enc = [enc.encode(v) for v, enc in zip(velocity, self.motor_encoders)]
        self.motor_sdr.dense = np.concatenate([sdr.dense for sdr in v_enc])
        return self.motor_sdr

    def reset_gaze_tracking(self):
        """
        Discard any prior gaze tracking.  Call this after forcibly moving eye
        to a new starting position.
        """
        self.gaze = [(
            self.eye_sensor.position[0],
            self.eye_sensor.position[1],
            self.eye_sensor.orientation,
            self.eye_sensor.scale)]

    def gaze_tracking(self, diag=True):
        """
        Returns vector of tuples of (position-x, position-y, orientation, scale)
        """
        if diag:
            im   = PIL.Image.fromarray(self.eye_sensor.image)
            draw = PIL.ImageDraw.Draw(im)
            width, height = im.size
            # Draw a red line through the centers of each gaze point
            for p1, p2 in zip(self.gaze, self.gaze[1:]):
                x1, y1, a1, s1 = p1
                x2, y2, a2, s2 = p2
                draw.line((y1, x1, y2, x2), fill='black', width=5)
                draw.line((y1, x1, y2, x2), fill='red', width=2)
            # Draw the bounding box of the eye sensor around each gaze point
            for x, y, orientation, scale in self.gaze:
                # Find the four corners of the eye's window
                corners = []
                for ec_x, ec_y in [(0,0), (0,-1), (-1,-1), (-1,0)]:
                    corners.append(self.eye_sensor.eye_coords[:, ec_x, ec_y])
                # Convert from list of pairs to index array.
                corners = np.transpose(corners)
                # Rotate the corners
                c = math.cos(orientation)
                s = math.sin(orientation)
                rot = np.array([[c, -s], [s, c]])
                corners = np.matmul(rot, corners)
                # Scale/zoom the corners
                corners *= scale
                # Position the corners
                corners += np.array([x, y]).reshape(2, 1)
                # Convert from index array to list of coordinates pairs
                corners = list(tuple(coord) for coord in np.transpose(corners))
                # Draw the points
                for start, end in zip(corners, corners[1:] + [corners[0]]):
                    line_coords = (start[1], start[0], end[1], end[0],)
                    draw.line(line_coords, fill='green', width=2)
            del draw
            plt.figure("Gaze Tracking")
            im = np.array(im)
            plt.imshow(im, interpolation='nearest')
            plt.show()
        return self.gaze[:]


def small_random_movement(eye_sensor):
    max_change_angle        = (2*3.14159) / 500
    eye_sensor.position     = (
        eye_sensor.position[0] + random.gauss(1, .75),
        eye_sensor.position[1] + random.gauss(1, .75),)
    eye_sensor.orientation += random.uniform(-max_change_angle, max_change_angle)
    eye_sensor.scale = 1


if __name__ == '__main__':
    eye = Eye()

    import datasets, random
    # data = datasets.Dataset('./datasets/small_items')
    data = datasets.Dataset('./datasets/textures')
    print("Num Images:", len(data))
    data.shuffle()
    for z in range(len(data)):
        eye.reset()
        data.next_image()
        img_path = data.current_image
        print("Loading image %s"%img_path)
        img = np.asarray(PIL.Image.open(img_path))
        eye.new_image(img)
        eye.scale = 1

        for i in range(10):
            sdr = eye.compute()
            eye.show_view()
            small_random_movement(eye)

    print("All images seen.")
