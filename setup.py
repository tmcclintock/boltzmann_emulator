from setuptools import setup

dist = setup(name="boltzmann_emulator",
             author="Thomas McClintock",
             author_email="mcclintock@bnl.gov",
             description="Framework for emulating boltzmann codes.",
             license="MIT",
             url="https://github.com/tmcclintock/boltzmann_emulator",
             packages=['boltzmann_emulator'],
             long_description=open("README.md").read()
)
