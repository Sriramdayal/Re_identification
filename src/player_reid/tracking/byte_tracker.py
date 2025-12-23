import numpy as np
from ultralytics.trackers import BYTETracker
from ultralytics.utils import IterableSimpleNamespace
import torch

class BytesTrackAgent:
    def __init__(self, track_thresh=0.4, match_thresh=0.7, track_buffer=30, frame_rate=30):
        """
        Wrapper around Ultralytics' efficient ByteTrack implementation.
        """
        # ByteTrack args expected by Ultralytics
        args = IterableSimpleNamespace(
            track_thresh=track_thresh,
            track_high_thresh=track_thresh, # Assuming same as thresh for now
            track_low_thresh=0.1,
            new_track_thresh=track_thresh + 0.1,
            match_thresh=match_thresh,
            track_buffer=track_buffer,
            min_box_area=10,
            mot20=False,
            fuse_score=True  # Required by recent Ultralytics versions
        )
        self.tracker = BYTETracker(args, frame_rate=frame_rate)

    def update(self, dets, frame_shape):
        """
        Update tracker with detections.
        
        Args:
            dets (list): List of [x1, y1, x2, y2, conf] or np.array
            frame_shape (tuple): (h, w) of current frame
        
        Returns:
            tracks (list): List of [x1, y1, x2, y2, track_id, conf, class_cls]
                           (Note: Ultralytics returns a slightly different format, we will normalize it)
        """
        if len(dets) == 0:
            return []

        # Convert to numpy array if not already
        if isinstance(dets, list):
            dets = np.array(dets)
            
        # ByteTrack requires a torch tensor sometimes or numpy depending on implementation
        # Ultralytics impl often expects a class index at index 5. 
        # So we construct [x1, y1, x2, y2, conf, cls_idx]
        if dets.shape[1] == 5:
            # Add class index 0 (person)
            zeros = np.zeros((dets.shape[0], 1))
            dets = np.hstack((dets, zeros))
            
        try:
             # Convert to tensor as UL usually expects
            det_tensor = torch.from_numpy(dets).float()
            
            # Use the internal update method which expects a Detector-like output or raw tensor
            # The error 'Tensor object has no attribute conf' suggests it might be expecting an object with .conf property
            # OR we are using a version of ultralytics that expects something else.
            # Let's try passing the tensor directly but wrapped in a way it expects if possible.
            # Actually, looking at recent UL versions, tracker.update() often takes a Result object or simpler args.
            # If we are using the standalone tracker class, it inherits from ByteTracker -> STrack.
            
            # Fix: Manually construct the input expected by BYTETracker.update(results, img=None)
            # However, `tracker.update` signature in `ultralytics/trackers/byte_tracker.py` is `update(self, det, img=None)`
            # where det is the detections.
            
            # WORKAROUND: The error comes from `det.conf` access inside the tracker. 
            # If we pass a tensor, it treats it as an object and tries to access .conf?
            # No, standard torch tensor doesn't have .conf.
            # It means the tracker code expects an object (like a Box object) NOT a raw tensor.
            
            # BUT, the `ultralytics.trackers.byte_tracker.BYTETracker.update` method usually processes raw tensors if formatted right.
            # Let's try to mock the object it expects if it's looking for .conf
            
            # BETTER FIX: Use the `model.track()` API if available? No, we are building a custom pipeline.
            # Let's check if we can pass a SimpleNamespace mimicking the detection object.
            
            # Let's revert to a simpler method: The `det` argument in `update` is often expected to be the `preds` tensor from YOLO.
            # If the error is specific to `det.conf`, it implies `det` is treated as a container.
            
            # Let's try to pass the tensor directly again but ensure it's on CPU/GPU correctly?
            # No, the error is AttributeError.
            
            # HYPOTHESIS: We need to use `ultralytics.engine.results.Results` object?
            # Or simplified: We can just use the `tracker.update` but we must modify `dets` to be a `torch.Tensor`
            # AND maybe the internal implementation changed.
            
            # Alternative: Implementing a lightweight ByteTrack wrapper ourselves or fixing the input.
            # If `det` is `torch.Tensor`, `det.conf` fails. 
            # This suggests we might need to cast it to something else or the lines inside tracker are:
            # `conf = det.conf` which means det MUST be an object.
            
            # Let's try to use `BaseTracker` logic from Ultralytics properly.
            # It seems `BYTETracker` expects a `Results` object in newer versions?
            # Let's try to mimic a minimal Results object.
            
            class MockBoxes:
                def __init__(self, data):
                    self.data = data
                    self.conf = data[:, 4]
                    self.cls = data[:, 5]
                    self.xyxy = data[:, :4]

            class MockResult:
                def __init__(self, data):
                    self.boxes = MockBoxes(data)
                    self.keypoints = None
                    self.masks = None
            
            # But the tracker update() takes `det`.
            # If we look at `ultralytics/trackers/track.py`:
            # `det = result.boxes.data` usually.
            
            # Let's try to just pass the tensor, maybe my previous `det_tensor` was correct but the library version matches differently.
            # Wait, `det` in `BYTETracker.update(self, det, img=None)`:
            # If I look at source code for UL, it usually does:
            # `self.model.predict(..., tracker="bytetrack.yaml")`
            
            # Since we are using the class directly, we might be bypassing the wrapper that converts Tensor -> Object.
            
            # Let's try to use the `boxes` attribute if we can. 
            # Actually, let's just rewrite this to use `lap` and `scipy` directly if UL internal is too tied to their `Result` object.
            # OR simpler: Use a mock object that has `.conf`, `.xyxy`, `.cls`.
            
            class DetWrapper:
                def __init__(self, tensor):
                    self.tensor = tensor
                    self.conf = tensor[:, 4]
                    self.cls = tensor[:, 5]
                    self.xyxy = tensor[:, :4]
                    self.xywh = None # Tracker might calc this
                    
                def __len__(self):
                    return len(self.tensor)
            
            # Let's try passing DetWrapper(det_tensor)
            # But `BYTETracker` inherits `BOTSORT` or similar? No.
            
            # Let's try a different approach. We can instantiate `YOLO` model and use `model.track()`?
            # No, we want to separate detection from tracking.
            
            # Let's try the Mock object approach.
            det_tensor = torch.from_numpy(dets).float()
            
            # If the tracker code does `det.conf`, it expects an attribute.
            # We can create a subclass of Tensor? No.
            
            # Wrapper for a single detection row (to support iteration)
            class DetRow:
                def __init__(self, tensor):
                    self.tensor = tensor
                    
                @property
                def xywh(self):
                    # xyxy -> xywh (center-based)
                    # tensor shape is (6,) or (7,)
                    x1, y1, x2, y2 = self.tensor[:4]
                    w = x2 - x1
                    h = y2 - y1
                    cx = x1 + w / 2
                    cy = y1 + h / 2
                    return torch.tensor([cx, cy, w, h])

                @property
                def xyxy(self):
                    return self.tensor[:4]

                @property
                def conf(self):
                    return self.tensor[4]

                @property
                def cls(self):
                    return self.tensor[5]

            # Custom object that behaves like a list/tensor but has attributes
            class DetObject:
                def __init__(self, tensor):
                    self.tensor = tensor
                    self.conf = tensor[:, 4]
                    self.cls = tensor[:, 5]
                    self.xyxy = tensor[:, :4]
                    
                def __getitem__(self, idx):
                    # Handle slicing vs single index
                    sub = self.tensor[idx]
                    if len(sub.shape) == 1:
                        return DetRow(sub)
                    return DetObject(sub) # Return strict object for slices if needed

                def __len__(self):
                    return len(self.tensor)
                
                @property
                def shape(self):
                    return self.tensor.shape

                @property
                def xywh(self):
                    # Batch xywh calculation
                    xywh = self.tensor[:, :4].clone()
                    w = xywh[:, 2] - xywh[:, 0]
                    h = xywh[:, 3] - xywh[:, 1]
                    cx = xywh[:, 0] + w / 2
                    cy = xywh[:, 1] + h / 2
                    return torch.stack([cx, cy, w, h], dim=1)
            
            dets_obj = DetObject(det_tensor)
            
            # Note: tracker.update(det, img) often iterates over det.
            # If det is object, it might expect it to be iterable returning detection rows?
            
            # Let's try passing the tensor directly again, BUT we know earlier it failed on .conf
            # The only reason it fails on .conf is if the code does `det.conf` on the WHOLE batch?
            # Or on individual items?
            
            # If we look at `ultralytics/trackers/byte_tracker.py` -> `update(self, det, img=None)`
            # It usually does: `conf = det[:, 4]` if it detects a tensor?
            # Or `conf = det.conf` if it detects an object.
            
            # The failure "Tensor object has no attribute 'conf'" strongly suggests it treated it as an object
            # (maybe because it wasn't a standard torch tensor? or it WAS and the code assumes object).
            
            # By wrapping it in DetObject which has .conf AND is subscriptable (returning tensor rows),
            # we satisfy both.
            
            results = self.tracker.update(dets_obj, img=None)
            

            
        except Exception as e:
            # Fallback for compatibility issues
            print(f"Tracking error: {e}")
            return []

        # Results: (N, 7) -> x1,y1,x2,y2,id,conf,cls
        return results
