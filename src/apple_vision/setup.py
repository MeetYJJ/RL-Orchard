from setuptools import setup

package_name = 'apple_vision'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/apple_detector.launch.py']),
        ('share/' + package_name + '/config', [
            'config/apple_detector.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='YOLO, ByteTrack and RealSense apple position publisher.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'apple_detector = '
            'apple_vision.apple_detector_node:main',
            'apple_listener = '
            'apple_vision.apple_listener:main',
        ],
    },
)
