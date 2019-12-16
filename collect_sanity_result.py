import sys, os
from multiprocessing import Queue, Process

if __name__=='__main__':
    # Plot learning curve + value along trajectory
    # Save the video of 5 test trajectories
    # Plot value vs gripper position.
    if len(sys.argv) < 2:
        print('Usage: python collect_sanity_result.py [dir_name]<xxx/her_augment_filter>')
    dir_name = sys.argv[1]
    # model_idx = int(sys.argv[2])
    folders = os.listdir(dir_name)
    commands = []
    for folder in folders:
        if not os.path.isdir(os.path.join(dir_name, folder)):
            continue
        files = os.listdir(os.path.join(dir_name, folder))
        model_idx = [int(f.split('_')[1].strip('.zip')) for f in files if f[:6] == 'model_']
        model_idx = max(model_idx)
        print('Model index', model_idx)
        if os.path.exists(os.path.join(dir_name, folder, 'model_%d.zip' % model_idx)):
            if os.path.exists(os.path.join(dir_name, folder, 'learning_curve.png')):
                os.remove(os.path.join(dir_name, folder, 'learning_curve.png'))
                print('Remove existing', os.path.join(dir_name, folder, 'learning_curve.png'))
            if os.path.exists(os.path.join(dir_name, folder, 'FetchPushWallObstacle-v4.mp4')):
                os.remove(os.path.join(dir_name, folder, 'FetchPushWallObstacle-v4.mp4'))
                print('Remove existing', os.path.join(dir_name, folder, 'FetchPushWallObstacle-v4.mp4'))
            if os.path.exists(os.path.join(dir_name, folder, 'value_gripperpos.mp4')):
                os.remove(os.path.join(dir_name, folder, 'value_gripperpos.mp4'))
                print('Remove existing', os.path.join(dir_name, folder, 'value_gripperpos.mp4'))
            model_path = os.path.join(dir_name, folder, 'model_35.zip')
            command = 'python ~/projects/fetcher/plot_log.py --log_path %s' % os.path.join(dir_name, folder)
            commands.append(command)
            command = 'python ~/projects/fetcher/test_universal_sanity.py --env FetchPushWallObstacle-v4 ' \
                      '--load_path %s --play --export_gif && python ~/projects/fetcher/visualize_value_sanity.py %s' \
                      % (model_path, model_path)
            commands.append(command)
    print('Collected', len(commands), 'commands')

    def worker(input, output):
        for cmd  in iter(input.get, 'STOP'):
            os.system(cmd)
            output.put('done')

    NUMBER_OF_PROCESSES = 4

    # Create queues
    task_queue = Queue()
    done_queue = Queue()

    # Submit tasks
    for cmd in commands:
        task_queue.put(cmd)

    # Start worker processes
    for i in range(NUMBER_OF_PROCESSES):
        Process(target=worker, args=(task_queue, done_queue)).start()

    # Get and print results
    print('Unordered results:')
    for i in range(NUMBER_OF_PROCESSES):
        print('\t', done_queue.get())

    # Tell child processes to stop
    for i in range(NUMBER_OF_PROCESSES):
        task_queue.put('STOP')
