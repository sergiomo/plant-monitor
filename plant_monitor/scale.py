import click
import logging
import os
import pigpio
import time

import HX711

class ScaleTimeOutError(Exception):
    pass

class ScaleCalibrationError(Exception):
    pass

class SampleCalibrator:
    DEFAULT_MODEL_ORDER = 1  # Simple gain and offset error calibration
    
    DEFAULT_COEFFICIENTS = [
        -320.3996379390544,
        0.002454239986910253
    ]
    
    def __init__(self, calibration_data_file_path):
        self.calibration_data_file_path = calibration_data_file_path
        if calibration_data_file_path is not None:
            self.coefficients = self.load_coefficients_from_file()
        else:
            logging.info('Using default calibration coefficients.')
            self.coefficients = self.DEFAULT_COEFFICIENTS
        
    def calibrate_raw_value(self, raw_value):
        # y = c0  +  c1 * x  +  c2 * x^2  +  c3 * x^3  +  ...
        return sum([c * (raw_value ** idx) for idx, c in enumerate(self.coefficients)])
        
    def load_coefficients_from_file(self):
        with open(self.calibration_data_file_path) as input_file:
            data = input_file.read()
        return self._deserialize_coefficients(data)
        
    def _serialize_coefficients(self, coefficient_list):
        return '\n'.join([str(x) for x in coefficient_list])
        
    def _deserialize_coefficients(self, coefficient_string):
        return [float(x.strip()) for x in coefficient_string.strip().split('\n')]
    
    def calculate_calibration_constants(
        self, 
        raw_values, 
        real_values, 
        model_order=DEFAULT_MODEL_ORDER,
        output_file_path=None,
    ):
        # Import modules here as to not slow down app startup.
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures
        import numpy as np
        
        y = np.array(real_values)
        X = np.array([[x] for x in raw_values])
        
        polynomial_preproc = PolynomialFeatures(model_order)
        X_trans = polynomial_preproc.fit_transform(X)
        
        fitted_model = LinearRegression(fit_intercept=False).fit(X_trans, y)
        coefficients = fitted_model.coef_
        
        # Fit a model here
        if output_file_path is not None:
            with open(output_file_path, 'w') as output_file:
                output_file.write(self._serialize_coefficients(coefficients))
        else:
            print('The calibration coefficients are:\n%s' % self._serialize_coefficients(coefficients))
            
        self.coefficients = coefficients


class HX711Scale:

    DEFAULT_SAMPLE_AVERAGING_WINDOW_SIZE = 5
    SAMPLE_BUFFER_SIZE = 1000
    DEFAULT_HX711_MODE = HX711.CH_A_GAIN_128
    
    # DEFAULT_LOAD_CELL_MAX_CAPACITY_GRAMS = 5000
    # HX711_MAX_VALUE = 0x7FFFFF # Decimal 8,388,607
    # HX711_MIN_VALUE = -1 * (HX711_MAX_VALUE + 1)  # Decimal -8,388,608, 0x800000 in twos complements
    HX711_MAX_SAMPLE_RATE = 80
    
    
    def __init__(
        self,
        hx711_clock_pin_number, 
        hx711_data_pin_number,
        calibration_data_file_path=None,
        hx711_channel_and_gain=DEFAULT_HX711_MODE,
        sample_averaging_window_size=DEFAULT_SAMPLE_AVERAGING_WINDOW_SIZE,
    ):
        self.hx711_clock_pin_number = hx711_clock_pin_number
        self.hx711_data_pin_number = hx711_data_pin_number
        self.hx711_channel_and_gain = hx711_channel_and_gain
        self.sample_averaging_window_size = sample_averaging_window_size
        
        self.sample_calibrator = SampleCalibrator(
            calibration_data_file_path=calibration_data_file_path
        )
        
        self.sensor_data = [None] * self.SAMPLE_BUFFER_SIZE
        self.most_recent_sample_number = 0
        self.hx711_mode_readback = None
        self.pigpio_client = None
        self.hx711_client = None
        
        self.started = False

    def _callback(self, count, mode, reading):
        """The pigpio library will spawn a thread that will wait for DIO data and then call
        this function."""
        self.most_recent_sample_number = count
        hx711_mode_readback = mode
        self.sensor_data = self.sensor_data[1:] + [reading]

    def start(self):
        if self.started:
            return
        self.pigpio_client = pigpio.pi()
        if not self.pigpio_client.connected:
            raise Exception(
            'Could not connect to the pigpio daemon. To start the daemon, run "sudo pigpiod".'
            )
        self.hx711_client = HX711.sensor(
            self.pigpio_client,
            DATA=self.hx711_data_pin_number,
            CLOCK=self.hx711_clock_pin_number,
            mode=self.hx711_channel_and_gain,
            callback=self._callback
        )
        # Give the hx711 library enough time to receive SETTLE_READINGS samples.
        while self.most_recent_sample_number < HX711.SETTLE_READINGS:
            time.sleep(1 / self.HX711_MAX_SAMPLE_RATE)
        self.started = True
    
    def get_sample(self, timeout_seconds=15, return_raw_value=False):
        if not self.started:
            self.start()
        num_past_samples = self.most_recent_sample_number
        target_sample_count = num_past_samples + self.sample_averaging_window_size
        timeout_at = time.time() + timeout_seconds
        timed_out = False
        # Note that self.most_recent_sample_number is written by a thread, and read by another.
        # This operation is thread-safe in Python, so no worries!
        while self.most_recent_sample_number < target_sample_count:
            if time.time() > timeout_at:
                timed_out = True
                break
            time.sleep(1 / self.HX711_MAX_SAMPLE_RATE)
        if timed_out:
            num_samples_collected = self.most_recent_sample_number - num_past_samples
            raise ScaleTimeOutError(
            'Timed out waiting for %d samples. Collected %d.' % (
                self.sample_averaging_window_size,
                self.num_samples_collected
            ))
        else:
            # Return the average of the last n samples
            raw_values = self.sensor_data[-self.sample_averaging_window_size:]
            if return_raw_value:
                # Raw values are integers.
                return int(sum(raw_values) / self.sample_averaging_window_size)
            else:
                calibrated_values = [self.sample_calibrator.calibrate_raw_value(x) for x in raw_values]
                return sum(calibrated_values) / len(calibrated_values)
    
    def stop(self):
        if not self.started:
            return
        self.hx711_client.pause()
        self.hx711_client.cancel()
        self.pigpio_client.stop()
        self.started = False
        
    def interactive_calibration(self, output_file_path, model_order=1):
        # TODO: Help the user set the right channel gain.

        raw_values = []
        real_weights = []
        keep_going = True
        
        print('-' * 80)
        print('Calibration procedure')
        print('-' * 80)
        
        required_samples = model_order + 1
        print('We\'ll need a total of %d data samples.' % required_samples)
        _ = input('First, make sure there\'s nothing on the scale and press [Enter]: ')
        raw_values.append(self.get_sample(return_raw_value=True))
        real_weights.append(0.0)
        first_time = True
        
        while keep_going:
            first_time_message = \
                'Now place an object whose weight you know on the scale, type the object\'s weight in grams and press [Enter]:\n'
            successive_times_message = \
                'Now place another object whose weight you know on the scale, type the object\'s weight in grams and press [Enter]:\n'
            enough_samples_message = \
                'We already have enough samples to calculate the calibration coefficients.' + \
                ' However, you can keep entering more data if you want a slightly more precise calibration.' + \
                ' To add more data, place an object whose weight you know on the scale, type the object\'s weight in grams and press [Enter].' + \
                ' Alternatively, enter "stop" to finish the calibration procedure:\n'

            if first_time:
                message = first_time_message
            elif len(real_weights) < required_samples:
                message = successive_times_message
            else:
                message = enough_samples_message
            response = input('\n' + message).strip().lower()
            if response.startswith('s'):
                break
            try:
                real_weights.append(float(response))
                raw_values.append(self.get_sample(return_raw_value=True))
                first_time = False
            except ValueError:
                print('Invalid input! Please type a weight in grams, or "stop" to finish.')
        if len(real_weights) < required_samples:
            raise ScaleCalibrationError(
                'You need to enter at least %d samples to calibrate a model of order %d.' % (
                    required_samples,
                    model_order
                ))
        print('Fitting calibration model...')
        self.sample_calibrator.calculate_calibration_constants(
            raw_values=raw_values, 
            real_values=real_weights, 
            model_order=model_order,
            output_file_path=output_file_path,
        )


DATA_GPIO_PIN_NUMBER = 5
CLOCK_GPIO_PIN_NUMBER = 6
DEFAULT_CALIBRATION_DATA_FILE_PATH = 'hx711_cal.txt'


@click.command()
@click.option(
    '-d', 
    '--data', 
    'data_pin_number', 
    type=int, 
    default=DATA_GPIO_PIN_NUMBER, 
    help='GPIO pin number connected to the HX711\'s data pin.'
)
@click.option(
    '-c', 
    '--clock', 
    'clock_pin_number', 
    type=int, 
    default=CLOCK_GPIO_PIN_NUMBER, 
    help='GPIO pin number connected to the HX711\'s data pin.'
)
@click.option(
    '-m',
    '--model-order',
    'calibration_model_order',
    type=int,
    default=1,
    help='The degree of the polynomial to use for calibration.'
)
@click.argument('operation', type=click.Choice(['calibration', 'live-scale'], case_sensitive=False))
def cli(data_pin_number, clock_pin_number, calibration_model_order, operation):
    if os.path.isfile(DEFAULT_CALIBRATION_DATA_FILE_PATH):
        calibration_data_file_path = DEFAULT_CALIBRATION_DATA_FILE_PATH
    else:
        logging.warning(
            'Calibration data file not found. Using default calibration coefficients.' + \
            ' Please run the calibration command.'
        )
        calibration_data_file_path = None
    my_scale = HX711Scale(
        hx711_clock_pin_number=clock_pin_number, 
        hx711_data_pin_number=data_pin_number,
        calibration_data_file_path=calibration_data_file_path,
    )
    if operation == 'calibration':
        calibration(my_scale, calibration_model_order)
    elif operation == 'live-scale':
        live_scale(my_scale)
    else:
        raise Exception('Unimplemented operation "%s"' % str(operation))

def calibration(my_scale, calibration_model_order):
    my_scale.start()
    my_scale.interactive_calibration(
        output_file_path=DEFAULT_CALIBRATION_DATA_FILE_PATH,
        model_order=calibration_model_order,
    )
    my_scale.stop()
    
def live_scale(my_scale):
    print('Live scale mode')
    my_scale.start()
    print('\nPress Ctrl-C at any time to exit.\n\n')
    try:
        while True:
            print('\033[F\33[2KWeight: %.3fg' % my_scale.get_sample())
    except KeyboardInterrupt:
        pass
    my_scale.stop()

if __name__ == '__main__':
    cli()
