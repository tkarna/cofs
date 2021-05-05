pipeline {
    agent {
      docker {
        image 'firedrakeproject/firedrake-vanilla:latest'
        label 'firedrakeproject'
        args '-v /var/run/docker.sock:/var/run/docker.sock'
        alwaysPull true
      }
    }
    environment {
        PATH = "/usr/local/bin:/usr/bin:/bin"
        PETSC_CONFIGURE_OPTIONS="--download-mumps --download-scalapack --download-parmetis --download-metis"
        CC = "mpicc"
    }
    stages {
        stage('Clean') {
            steps {
                dir('build') {
                    deleteDir()
                }
            }
        }
        stage('Update Firedrake') {
            steps {
                sh 'mkdir build'
                dir('build') {
                    timestamps {
                        sh '''
sudo -u firedrake /bin/bash << Here
whoami
cd /home/firedrake
. /home/firedrake/firedrake/bin/activate
firedrake-update || (cat firedrake-update.log && /bin/false)
install -d /home/firedrake/firedrake/.cache
chmod -R a+rwx /home/firedrake/firedrake/.cache
firedrake-status
Here
'''
                    }
                }
            }
        }
        stage('Install Thetis') {
            steps {
                timestamps {
                    sh '''
. /home/firedrake/firedrake/bin/activate
sudo -u firedrake -H /home/firedrake/firedrake/bin/python -m pip install -r requirements.txt
sudo -u firedrake -H chmod a+rwx /home/firedrake/firedrake/lib/python*/site-packages
sudo -u firedrake -H chmod a+rwx /home/firedrake/firedrake/lib/python*/site-packages/easy-install.pth
sudo -u firedrake -H chmod a+rwx /home/firedrake/firedrake/bin
chmod a+w .
/home/firedrake/firedrake/bin/python -m pip install -e .
'''
                }
            }
        }
        stage('Test') {
            parallel {
                stage('Forward Test') {
                    steps {
                        timestamps {
                            sh '''
. /home/firedrake/firedrake/bin/activate
python -mpytest -v test/ -n 11
'''
                       }
                    }
                }
                stage('Test Adjoint') {
                    steps {
                        timestamps {
                            sh '''
. /home/firedrake/firedrake/bin/activate
python -mpytest -v test_adjoint/
'''
                        }
                    }
                }
            }
        }
        stage('Lint') {
            steps {
                timestamps {
                    sh '''
. /home/firedrake/firedrake/bin/activate
make lint
'''
                }
            }
        }
    }
}
