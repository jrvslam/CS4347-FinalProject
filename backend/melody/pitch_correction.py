def perform_pitch_correction(annotations, threshold=0.2):
    corrected_annotations = []
    i = 0
    while i < len(annotations):
        annotation = annotations[i]
        onset_time = annotation[0]
        offset_time = annotation[1]
        pitch = annotation[2]
        duration = offset_time - onset_time
        # Perform correction if duration of annotation is too short
        if (duration < threshold):
            prev_annotation = [0, 0, 0]
            next_annotation = [0, 0, 0]
            if (len(corrected_annotations) > 0):
                prev_annotation = corrected_annotations[-1]
            if (i + 1 < len(annotations)):
                next_annotation = annotations[i+1]
            prev_annotation_pitch = prev_annotation[2]
            next_annotation_pitch = next_annotation[2]
            if (onset_time == prev_annotation[1] and prev_annotation_pitch == pitch and offset_time == next_annotation[0] and next_annotation_pitch == pitch):
                # If pitch is identical to previous and next annotation, merge all 3 annotations
                corrected_annotations[-1] = [prev_annotation[0], next_annotation[1], pitch]
                i = i + 2
            elif (onset_time == prev_annotation[1] and prev_annotation_pitch == pitch):
                # If pitch is identical to previous annotation, merge with previous annotation
                corrected_annotations[-1] = [prev_annotation[0], offset_time, pitch]
                i = i + 1
            elif (offset_time == next_annotation[0] and next_annotation_pitch == pitch):
                # If pitch is identical to next annotation, merge with next annotation
                corrected_annotations.append([onset_time, next_annotation[1], pitch])
                i = i + 2
            else:
                # Drop annotation
                i = i + 1
        else:
            corrected_annotations.append(annotation)
            i = i + 1
        
    return corrected_annotations